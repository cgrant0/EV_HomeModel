import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class BatteryOptimizer:
    """
    Optimizes the usage of a battery (e.g., EV battery) to minimize energy costs
    and provide backup during outages.
    """
    
    def __init__(self, battery_capacity=60, efficiency=0.9, max_charge_rate=11, max_discharge_rate=11):
        """
        Initialize the battery optimizer
        
        Parameters:
        battery_capacity (float): Battery capacity in kWh
        efficiency (float): Round-trip efficiency of the battery (0-1)
        max_charge_rate (float): Maximum charging rate in kW
        max_discharge_rate (float): Maximum discharging rate in kW
        """
        self.battery_capacity = battery_capacity
        self.efficiency = efficiency
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        
        # Default values
        self.hourly_rates = None
        self.initial_soc = 0.2  # Start at 20% state of charge
        self.min_soc = 0.2      # Minimum state of charge (for backup)
        self.max_soc = 0.9      # Maximum state of charge (for battery health)
        
        # For keeping track of cycles
        self.total_energy_charged = 0
        self.cycle_count = 0
    
    def set_price_schedule(self, hourly_rates):
        """
        Set the hourly electricity rates
        
        Parameters:
        hourly_rates (dict or pd.Series): Hourly electricity rates ($/kWh)
        """
        if isinstance(hourly_rates, dict):
            self.hourly_rates = hourly_rates
        elif isinstance(hourly_rates, pd.Series):
            self.hourly_rates = hourly_rates.to_dict()
        else:
            raise ValueError("hourly_rates must be a dictionary or pandas Series")
    
    def set_battery_params(self, initial_soc=0.2, min_soc=0.2, max_soc=0.9):
        """
        Set battery parameters
        
        Parameters:
        initial_soc (float): Initial state of charge (0-1)
        min_soc (float): Minimum allowed state of charge (0-1)
        max_soc (float): Maximum allowed state of charge (0-1)
        """
        self.initial_soc = min(max(initial_soc, 0), 1)
        self.min_soc = min(max(min_soc, 0), 1)
        self.max_soc = min(max(max_soc, 0), 1)
        
        # Reset cycle tracking
        self.total_energy_charged = 0
        self.cycle_count = 0
    
    def optimize_battery_usage(self, hourly_consumption, ev_availability=None, outage_prob=None):
        """
        Optimize battery usage based on hourly consumption, electricity rates,
        EV availability, and outage probability.
        
        Parameters:
        hourly_consumption (list or pd.Series): Hourly energy consumption in kWh
        ev_availability (list or pd.Series, optional): Boolean or 0-1 indicating when EV is available
        outage_prob (list or pd.Series, optional): Hourly outage probability (0-1)
        
        Returns:
        dict: Results containing battery actions and state at each hour
        """
        if self.hourly_rates is None:
            raise ValueError("Hourly rates not set. Call set_price_schedule() first.")
        
        # Ensure data is in the right format
        if isinstance(hourly_consumption, pd.Series):
            hourly_consumption = hourly_consumption.values
            
        if ev_availability is None:
            # Default: EV available all day
            ev_availability = [1] * 24
        elif isinstance(ev_availability, pd.Series):
            ev_availability = ev_availability.values
            
        if outage_prob is None:
            # Default: No outage probability
            outage_prob = [0] * 24
        elif isinstance(outage_prob, pd.Series):
            outage_prob = outage_prob.values
        
        # Initialize results
        results = {
            'hour': list(range(24)),
            'consumption': hourly_consumption,
            'rate': [self.hourly_rates.get(h, 0) for h in range(24)],
            'ev_available': ev_availability,
            'battery_charge': [0] * 24,  # Positive: charging, Negative: discharging
            'battery_soc': [self.initial_soc] * 24,
            'grid_consumption': [0] * 24,
            'cost': [0] * 24,
            'outage_protected': [False] * 24
        }
        
        # Initial state of charge
        soc = self.initial_soc
        usable_capacity = self.battery_capacity * (self.max_soc - self.min_soc)
        
        # Identify best hours to charge based on rates
        hourly_rate_items = [(hour, rate) for hour, rate in self.hourly_rates.items() if hour < 24]
        sorted_hours = sorted(hourly_rate_items, key=lambda x: x[1])
        
        # First, allocate capacity to protect against outages if probability is significant
        reserved_capacity = {}
        for hour in range(24):
            if outage_prob[hour] > 0.1:  # Threshold for considering outage protection
                # Reserve enough capacity to cover consumption during outage
                energy_needed = hourly_consumption[hour]
                reserved_capacity[hour] = energy_needed
        
        # Simulate the day hour by hour
        for hour in range(24):
            # Check if EV is available
            if not ev_availability[hour]:
                # EV not available, battery can't be used
                results['battery_charge'][hour] = 0
                results['grid_consumption'][hour] = hourly_consumption[hour]
                results['battery_soc'][hour] = soc
                results['cost'][hour] = hourly_consumption[hour] * results['rate'][hour]
                continue
            
            # Calculate available energy from battery (considering efficiency for discharge)
            available_energy = (soc - self.min_soc) * self.battery_capacity
            
            # Check for outage protection need
            if hour in reserved_capacity and reserved_capacity[hour] > 0:
                # Reserve capacity for outage
                protection_capacity = min(available_energy, reserved_capacity[hour])
                available_energy -= protection_capacity
                results['outage_protected'][hour] = True
            
            # Determine whether to charge or discharge based on rate
            current_rate = results['rate'][hour]
            
            # Find the position of the current hour in the sorted rates
            rate_rank = next((i for i, (h, r) in enumerate(sorted_hours) if h == hour), len(sorted_hours)-1)
            
            # Charge during cheap hours (lower third of rates)
            if rate_rank < len(sorted_hours) // 3:
                # Calculate how much the battery can be charged
                space_available = (self.max_soc - soc) * self.battery_capacity
                max_charge = min(space_available, self.max_charge_rate)
                
                # Charge the battery
                charge_amount = max_charge * self.efficiency
                soc += charge_amount / self.battery_capacity
                
                # Update results
                results['battery_charge'][hour] = charge_amount
                results['grid_consumption'][hour] = hourly_consumption[hour] + charge_amount
                self.total_energy_charged += charge_amount
            
            # Discharge during expensive hours (upper third of rates)
            elif rate_rank > 2 * len(sorted_hours) // 3 and available_energy > 0:
                # Calculate how much the battery can discharge
                max_discharge = min(available_energy, self.max_discharge_rate, hourly_consumption[hour])
                
                # Discharge the battery
                soc -= max_discharge / self.battery_capacity
                
                # Update results
                results['battery_charge'][hour] = -max_discharge
                results['grid_consumption'][hour] = hourly_consumption[hour] - max_discharge
            
            # Otherwise, neither charge nor discharge
            else:
                results['grid_consumption'][hour] = hourly_consumption[hour]
            
            # Update state of charge and cost for this hour
            results['battery_soc'][hour] = soc
            results['cost'][hour] = results['grid_consumption'][hour] * current_rate
        
        # Calculate cycle count (1 cycle = full battery capacity charged)
        self.cycle_count = self.total_energy_charged / self.battery_capacity
        
        return results
    
    def plot_results(self, results, show_plot=True, save_path=None):
        """
        Plot the battery optimization results
        
        Parameters:
        results (dict): Results from optimize_battery_usage()
        show_plot (bool): Whether to display the plot
        save_path (str, optional): Path to save the plot
        
        Returns:
        tuple: Figure and axes objects
        """
        # Create figure with proper spacing
        fig = plt.figure(figsize=(12, 15))
        gs = fig.add_gridspec(3, 1, hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        # Convert data to numpy arrays
        hours = np.array(results['hour'])
        consumption = np.array(results['consumption'])
        battery_charge = np.array(results['battery_charge'])
        
        # For battery discharge (negative values), show as positive contribution
        battery_discharge = np.array([-min(0, charge) for charge in battery_charge])
        
        # Plot 1: Consumption breakdown
        # Create bars with explicit labels and store references
        home_cons = ax1.bar(hours, consumption, color='skyblue', alpha=0.7, label='Home Consumption')
        
        # Add battery discharge bars
        has_discharge = False
        for h, discharge in enumerate(battery_discharge):
            if discharge > 0:
                if not has_discharge:
                    batt_discharge = ax1.bar(h, discharge, color='green', alpha=0.7, label='Battery Discharging')
                    has_discharge = True
                else:
                    ax1.bar(h, discharge, color='green', alpha=0.7)
        
        ax1.set_ylabel('Energy (kWh)')
        ax1.set_title('Hourly Energy Breakdown')
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Create legend with explicit handles
        legend_handles = [home_cons]
        legend_labels = ['Home Consumption']
        if has_discharge:
            legend_handles.append(batt_discharge)
            legend_labels.append('Battery Discharging')
        ax1.legend(legend_handles, legend_labels, loc='upper right')
        
        # Plot 2: Battery state of charge
        soc_pct = np.array([soc * 100 for soc in results['battery_soc']])
        soc_line = ax2.plot(hours, soc_pct, 'b-', linewidth=2, label='State of Charge')[0]
        ax2.fill_between(hours, self.min_soc * 100, soc_pct, alpha=0.3, color='blue')
        
        # Show EV availability
        availability = results['ev_available']
        has_unavailable = False
        ev_span = None
        for h, avail in enumerate(availability):
            if not avail:
                if not has_unavailable:
                    ev_span = ax2.axvspan(h-0.4, h+0.4, color='gray', alpha=0.3, label='EV Unavailable')
                    has_unavailable = True
                else:
                    ax2.axvspan(h-0.4, h+0.4, color='gray', alpha=0.3)
        
        # Highlight outage protected periods
        has_protected = False
        outage_span = None
        for h, protected in enumerate(results['outage_protected']):
            if protected:
                if not has_protected:
                    outage_span = ax2.axvspan(h-0.4, h+0.4, color='green', alpha=0.1, label='Outage Protected')
                    has_protected = True
                else:
                    ax2.axvspan(h-0.4, h+0.4, color='green', alpha=0.1)
        
        ax2.set_ylabel('State of Charge (%)')
        ax2.set_title('Battery State of Charge')
        ax2.set_ylim(0, 100)
        ax2.set_xlim(-0.5, 23.5)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Create legend with explicit handles
        legend_handles = [soc_line]
        legend_labels = ['State of Charge']
        if ev_span:
            legend_handles.append(ev_span)
            legend_labels.append('EV Unavailable')
        if outage_span:
            legend_handles.append(outage_span)
            legend_labels.append('Outage Protected')
        ax2.legend(legend_handles, legend_labels, loc='upper right')
        
        # Plot 3: Costs and rates
        hourly_costs = np.array(results['cost'])
        rates = np.array([results['rate'][h] for h in range(24)])
        
        # Create bars and line with explicit labels
        cost_bars = ax3.bar(hours, hourly_costs, color='indianred', alpha=0.7, label='Hourly Cost')
        ax3_rate = ax3.twinx()
        rate_line = ax3_rate.plot(hours, rates, 'g-', linewidth=2, label='Rate ($/kWh)')[0]
        
        ax3.set_xlabel('Hour of Day')
        ax3.set_xlim(-0.5, 23.5)
        ax3.set_xticks(range(0, 24, 2))
        ax3.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
        ax3.set_ylabel('Cost ($)')
        ax3_rate.set_ylabel('Rate ($/kWh)')
        ax3.set_title('Hourly Costs and Rates')
        ax3.grid(True, linestyle='--', alpha=0.3)
        
        # Create legend with explicit handles
        legend_handles = [cost_bars.patches[0], rate_line]
        legend_labels = ['Hourly Cost', 'Rate ($/kWh)']
        ax3.legend(legend_handles, legend_labels, loc='upper right')
        
        # Add summary information
        total_cost = sum(hourly_costs)
        total_consumption = sum(consumption)
        total_grid = sum(results['grid_consumption'])
        battery_contribution = total_consumption - total_grid
        
        summary = (
            f"Total Cost: ${total_cost:.2f}\n"
            f"Total Consumption: {total_consumption:.1f} kWh\n"
            f"Grid Consumption: {total_grid:.1f} kWh\n"
            f"Battery Contribution: {battery_contribution:.1f} kWh\n"
            f"Estimated Cycles: {self.cycle_count:.2f}"
        )
        
        # Add summary text with proper spacing
        fig.text(0.5, 0.02, summary, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout with explicit spacing
        fig.set_tight_layout(True)
        fig.subplots_adjust(bottom=0.1, hspace=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig, (ax1, ax2, ax3)
    
    def calculate_savings(self, results, baseline_cost=None):
        """
        Calculate savings from battery optimization
        
        Parameters:
        results (dict): Results from optimize_battery_usage()
        baseline_cost (float, optional): Baseline cost without battery optimization
        
        Returns:
        tuple: (savings_amount, savings_percentage)
        """
        # Calculate total cost with battery optimization
        optimized_cost = sum(results['cost'])
        
        # Calculate baseline cost if not provided
        if baseline_cost is None:
            baseline_cost = sum([results['consumption'][h] * results['rate'][h] for h in range(24)])
        
        # Calculate savings
        savings_amount = baseline_cost - optimized_cost
        savings_percentage = (savings_amount / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        return savings_amount, savings_percentage
    
    @classmethod
    def create_sample_price_schedules(cls):
        """
        Create sample price schedules for different utility rate structures
        
        Returns:
        dict: Dictionary of sample price schedules
        """
        # Flat rate
        flat_rate = {hour: 0.13 for hour in range(24)}
        
        # Time-of-use (peak/off-peak)
        tou_rate = {}
        for hour in range(24):
            if 8 <= hour < 12 or 17 <= hour < 21:  # Peak hours
                tou_rate[hour] = 0.25
            else:  # Off-peak hours
                tou_rate[hour] = 0.08
        
        # Time-of-use (peak/mid-peak/off-peak)
        tou_three_tier = {}
        for hour in range(24):
            if 17 <= hour < 21:  # Peak hours
                tou_three_tier[hour] = 0.30
            elif 8 <= hour < 17:  # Mid-peak hours
                tou_three_tier[hour] = 0.18
            else:  # Off-peak hours
                tou_three_tier[hour] = 0.07
        
        # Hourly dynamic pricing (sample)
        dynamic_rate = {
            0: 0.06, 1: 0.055, 2: 0.052, 3: 0.05, 4: 0.051, 5: 0.055,
            6: 0.075, 7: 0.11, 8: 0.15, 9: 0.18, 10: 0.21, 11: 0.23,
            12: 0.22, 13: 0.21, 14: 0.22, 15: 0.24, 16: 0.28, 17: 0.35,
            18: 0.42, 19: 0.38, 20: 0.32, 21: 0.25, 22: 0.15, 23: 0.10
        }
        
        return {
            'Flat Rate': flat_rate,
            'Time-of-Use (2-tier)': tou_rate,
            'Time-of-Use (3-tier)': tou_three_tier,
            'Dynamic Pricing': dynamic_rate
        } 