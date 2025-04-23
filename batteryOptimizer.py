import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates

CITY_DATA = {
    'Atlanta': {
        'dwellings': 449880,
        'weather_file': 'Atlanta/atlantaairportweather_72219013874.csv',
        'energy_file': 'Atlanta/Atlanta_15_minute_timeseries_data.csv'
    },
    'Ann Arbor': {
        'dwellings': 149394,
        'weather_file': 'Ann_Arbor/detroitairportweather_72537094847.csv',
        'energy_file': 'Ann_Arbor/AnnArbor_15_minute_timeseries_data_year.csv'
    },
    'San Diego': {
        'dwellings': 1187651,
        'weather_file': 'San_Diego/sandiegoairportweather_72290023188.csv',
        'energy_file': 'San_Diego/SanDiego_15_minute_timeseries_data.csv'
    }
}

class BatteryOptimizer:
    def __init__(self, hourly_rates, battery_capacity=60, efficiency=0.9):
        """
        Initialize the battery optimizer.
        
        Parameters:
        hourly_rates (list): List of 24 hourly electricity rates
        battery_capacity (float): Battery capacity in kWh
        efficiency (float): Round-trip efficiency of the battery (charge/discharge)
        """
        self.hourly_rates = hourly_rates
        self.battery_capacity = battery_capacity
        self.efficiency = efficiency
        self.discharge_efficiency = np.sqrt(efficiency)  # Discharge efficiency
        self.charge_efficiency = np.sqrt(efficiency)     # Charge efficiency
        
        # Hours of the day
        self.hours = list(range(24))
    
    def energy_consumption_from_temp(self, temperature):
        """
        Estimate hourly energy consumption based on temperature.
        Uses a quadratic model where consumption is related to temperature squared.
        
        Parameters:
        temperature (float): Temperature in Celsius
        
        Returns:
        float: Estimated hourly energy consumption in kWh
        """
        # Calculate consumption using quadratic model
        consumption = self.a * temperature**2 + self.b * temperature + self.c
        
        # Add some random variation (10% of base consumption)
        variation = np.random.normal(0, self.c * 0.1)
        consumption += variation
        
        # Ensure consumption is never negative
        return max(consumption, 0.1)
    
    def optimize_battery_usage(self, hourly_consumption):
        """
        Optimize battery charging/discharging to minimize cost using actual consumption data.
        
        Parameters:
        hourly_consumption (list): List of 24 hourly energy consumption values in kWh
        
        Returns:
        dict: Optimization results including savings and battery actions
        """
        # Total cost without battery
        baseline_cost = sum(consumption * rate for consumption, rate in zip(hourly_consumption, self.hourly_rates))
        
        # Simple greedy approach - charge during cheapest hours, discharge during most expensive
        
        # Prepare for tracking battery state
        battery_state = np.ones(25) * 60  # kWh in battery at the end of each hour (plus initial state)
        battery_charge = np.zeros(24)  # kWh charged during each hour
        battery_discharge = np.zeros(24)  # kWh discharged during each hour
        grid_consumption = hourly_consumption.copy()  # kWh drawn from grid
        
        # Sort hours by rate
        cheap_hours = sorted(range(24), key=lambda i: self.hourly_rates[i])
        expensive_hours = sorted(range(24), key=lambda i: self.hourly_rates[i], reverse=True)
        
        # First, determine how much energy we'll need for the expensive hours
        energy_needed = 0
        for hour in expensive_hours:
            # Only consider hours with rates higher than the cheapest rate
            if self.hourly_rates[hour] > self.hourly_rates[cheap_hours[0]]:
                energy_needed += min(hourly_consumption[hour], 10)  # Assume max 10kWh discharge per hour
                if energy_needed >= self.battery_capacity / self.discharge_efficiency:
                    break
        
        # First, charge battery during cheapest hours
        energy_to_charge = min(energy_needed / self.discharge_efficiency, self.battery_capacity)
        remaining_capacity = energy_to_charge
        
        for hour in cheap_hours:
            if remaining_capacity > 0:
                # Calculate maximum charging rate (assume max 10kWh per hour)
                max_charge = min(10, remaining_capacity)
                battery_charge[hour] = max_charge
                grid_consumption[hour] += max_charge / self.charge_efficiency  # Account for charging efficiency
                remaining_capacity -= max_charge
        
        # Calculate cumulative battery state
        for hour in range(24):
            battery_state[hour+1] = battery_state[hour] + battery_charge[hour] - battery_discharge[hour]
        
        # Then, discharge battery during most expensive hours
        for hour in expensive_hours:
            # Only discharge if the hour's rate is higher than the cheapest rate
            if self.hourly_rates[hour] > self.hourly_rates[cheap_hours[0]]:
                available_energy = battery_state[hour]
                needed_energy = hourly_consumption[hour]
                
                # Calculate maximum discharge amount (limited by battery state, consumption need, and max rate)
                max_discharge = min(available_energy, needed_energy, 10)
                
                if max_discharge > 0:
                    # Apply discharge efficiency
                    effective_discharge = max_discharge * self.discharge_efficiency
                    battery_discharge[hour] = max_discharge
                    grid_consumption[hour] -= effective_discharge
                    
                    # Update battery state for all subsequent hours
                    for h in range(hour+1, 25):
                        battery_state[h] -= max_discharge
        
        # Calculate new cost with battery
        optimized_cost = sum(consumption * rate for consumption, rate in zip(grid_consumption, self.hourly_rates))
        savings = baseline_cost - optimized_cost
        
        return {
            "baseline_consumption": hourly_consumption,
            "baseline_cost": baseline_cost,
            "optimized_consumption": grid_consumption,
            "optimized_cost": optimized_cost,
            "savings": savings,
            "battery_charge": battery_charge,
            "battery_discharge": battery_discharge,
            "battery_state": battery_state[:-1],  # Remove the extra element for initial state
            "hourly_rates": self.hourly_rates
        }
    
    def plot_results(self, results, date_str=None):
        """
        Plot the optimization results.
        
        Parameters:
        results (dict): Results from optimize_battery_usage
        date_str (str): Date string for the title (optional)
        """
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16))
        
        title_suffix = f" - {date_str}" if date_str else ""
        
        # Plot 1: Electricity rates
        ax1.bar(self.hours, results["hourly_rates"], color='green', alpha=0.7)
        ax1.set_title(f'Hourly Electricity Rates{title_suffix}')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Rate ($/kWh)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Energy consumption comparison
        width = 0.35
        ax2.bar(np.array(self.hours) - width/2, results["baseline_consumption"], width, label='Without Battery', color='red', alpha=0.6)
        ax2.bar(np.array(self.hours) + width/2, results["optimized_consumption"], width, label='With Battery', color='blue', alpha=0.6)
        ax2.set_title(f'Energy Consumption Comparison{title_suffix}')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Energy (kWh)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Battery state
        ax3.plot(self.hours, results["battery_state"], 'b-', label='Battery State')
        ax3.set_title('Battery State')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Energy (kWh)')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 4: Battery charging and discharging
        ax4.bar(self.hours, results["battery_charge"], color='green', alpha=0.7, label='Charging')
        ax4.bar(self.hours, -results["battery_discharge"], color='red', alpha=0.7, label='Discharging')
        ax4.set_title('Battery Operation')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Energy (kWh)')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"BATTERY OPTIMIZATION SUMMARY{title_suffix}")
        print("="*50)
        print(f"Total consumption: {sum(results['baseline_consumption']):.2f} kWh")
        print(f"Cost without battery: ${results['baseline_cost']:.2f}")
        print(f"Cost with battery: ${results['optimized_cost']:.2f}")
        print(f"Daily savings: ${results['savings']:.2f}")
        print(f"Annual savings (if similar pattern daily): ${results['savings'] * 365:.2f}")
        print("="*50)


def clean_weather_data(weather_df):
    """
    Clean and process weather data from SYNOP format.
    
    Parameters:
    weather_df (DataFrame): Raw weather data
    
    Returns:
    DataFrame: Cleaned weather data with datetime and temperature_c columns
    """
    # Convert temperature from SYNOP format to Celsius
    weather_df['temperature_c'] = weather_df['TMP'].apply(
        lambda x: (float(str(x)[1:-2]) / 10 * (-1 if str(x)[0] == '-' else 1)) if pd.notnull(x) else np.nan
    )
   
    # Remove rows where temperature_c is 999.9
    weather_df = weather_df.loc[weather_df['temperature_c'] != 999.9]
    
    # Convert to datetime
    weather_df['datetime'] = pd.to_datetime(weather_df['DATE']).dt.tz_localize(None)
    
    return weather_df[['datetime', 'temperature_c']].dropna()


def process_energy_data(energy_df, num_dwellings):
    """
    Process energy data to get per-dwelling consumption.
    
    Parameters:
    energy_df (DataFrame): Raw energy data
    num_dwellings (int): Number of dwellings for normalization
    
    Returns:
    DataFrame: Processed energy data with datetime and load_per_dwelling columns
    """
    # Convert timestamp to datetime
    energy_df['datetime'] = pd.to_datetime(energy_df['Timestamp (EST)']).dt.tz_localize(None)
   
    # Get total energy consumption
    energy_df['total_load'] = energy_df['baseline.out.electricity.total.energy_consumption.kwh']
   
    # Normalize per dwelling
    energy_df['load_per_dwelling'] = energy_df['total_load'] / num_dwellings
   
    return energy_df[['datetime', 'total_load', 'load_per_dwelling']]


def analyze_day(weather_df, energy_df, date, optimizer, plot=True):
    """
    Analyze a specific day for battery optimization.
    
    Parameters:
    weather_df (DataFrame): Cleaned weather data
    energy_df (DataFrame): Cleaned energy data
    date (str): Date to analyze in format 'YYYY-MM-DD'
    optimizer (BatteryOptimizer): Configured optimizer object
    plot (bool): Whether to plot results
    
    Returns:
    dict: Optimization results
    """
    # Convert date string to datetime
    date_obj = pd.to_datetime(date)
    next_day = date_obj + timedelta(days=1)
    
    # Filter data for the specified day
    day_energy = energy_df[(energy_df['datetime'] >= date_obj) & 
                           (energy_df['datetime'] < next_day)]
    
    if day_energy.empty:
        print(f"No energy data available for {date}")
        return None
    
    # Resample to hourly data
    hourly_energy = day_energy.set_index('datetime').resample('h').mean()
    
    # Get hourly consumption
    hourly_consumption = hourly_energy['load_per_dwelling'].values
    
    # If we have missing hours, fill them using linear interpolation
    if len(hourly_consumption) < 24:
        print(f"Warning: Only {len(hourly_consumption)} hours of energy data available for {date}.")
        print("Filling missing hours with linear interpolation.")
        
        # Create complete time index
        complete_index = pd.date_range(start=date_obj, end=next_day, freq='h', closed='left')
        
        # Create series with complete index
        complete_series = pd.Series(index=complete_index)
        
        # Fill available data
        for idx, value in zip(hourly_energy.index, hourly_consumption):
            complete_series[idx] = value
        
        # Interpolate missing values
        complete_series = complete_series.interpolate(method='linear')
        
        hourly_consumption = complete_series.values
    
    # Run optimization
    results = optimizer.optimize_battery_usage(hourly_consumption)
    
    # Plot if requested
    if plot:
        optimizer.plot_results(results, date)
    
    return results


def compare_temperature_impact(weather_df, energy_df, base_date, hot_date, peak_optimizer, offpeak_optimizer):
    """
    Compare the impact of temperature difference between two days.
    
    Parameters:
    weather_df (DataFrame): Cleaned weather data
    energy_df (DataFrame): Cleaned energy data
    base_date (str): Cooler day date in format 'YYYY-MM-DD'
    hot_date (str): Warmer day date in format 'YYYY-MM-DD'
    optimizer (BatteryOptimizer): Configured optimizer object
    
    Returns:
    dict: Comparison results
    """
    # Analyze both days
    base_results = analyze_day(weather_df, energy_df, base_date, offpeak_optimizer, plot=False)
    hot_results = analyze_day(weather_df, energy_df, hot_date, peak_optimizer, plot=False)
    
    if base_results is None or hot_results is None:
        print("Cannot compare days due to missing data.")
        return None
    
    # Calculate temperature difference
    base_avg_temp = np.mean(base_results['hourly_temps'])
    hot_avg_temp = np.mean(hot_results['hourly_temps'])
    temp_difference = hot_avg_temp - base_avg_temp
    
    # Calculate consumption and cost differences
    base_consumption = sum(base_results['baseline_consumption'])
    hot_consumption = sum(hot_results['baseline_consumption'])
    consumption_difference = hot_consumption - base_consumption
    
    base_cost_no_battery = base_results['baseline_cost']
    hot_cost_no_battery = hot_results['baseline_cost']
    cost_difference_no_battery = hot_cost_no_battery - base_cost_no_battery
    
    base_cost_with_battery = base_results['optimized_cost']
    hot_cost_with_battery = hot_results['optimized_cost']
    cost_difference_with_battery = hot_cost_with_battery - base_cost_with_battery
    
    # Calculate savings
    base_savings = base_results['savings']
    hot_savings = hot_results['savings']
    additional_savings = hot_savings - base_savings
    
    # Create comparison report
    comparison = {
        'base_date': base_date,
        'hot_date': hot_date,
        'base_avg_temp': base_avg_temp,
        'hot_avg_temp': hot_avg_temp,
        'temp_difference': temp_difference,
        'base_consumption': base_consumption,
        'hot_consumption': hot_consumption,
        'consumption_difference': consumption_difference,
        'base_cost_no_battery': base_cost_no_battery,
        'hot_cost_no_battery': hot_cost_no_battery,
        'cost_difference_no_battery': cost_difference_no_battery,
        'base_cost_with_battery': base_cost_with_battery,
        'hot_cost_with_battery': hot_cost_with_battery,
        'cost_difference_with_battery': cost_difference_with_battery,
        'base_savings': base_savings,
        'hot_savings': hot_savings,
        'additional_savings': additional_savings
    }
    
    # Print comparison report
    print("\nTEMPERATURE IMPACT COMPARISON")
    print("="*50)
    print(f"Cooler day ({base_date}): Average temperature {base_avg_temp:.1f}°C")
    print(f"Warmer day ({hot_date}): Average temperature {hot_avg_temp:.1f}°C")
    print(f"Temperature difference: {temp_difference:.1f}°C")
    print("-"*50)
    print(f"Cooler day consumption: {base_consumption:.2f} kWh per dwelling")
    print(f"Warmer day consumption: {hot_consumption:.2f} kWh per dwelling")
    print(f"Additional consumption due to temperature: {consumption_difference:.2f} kWh")
    print("-"*50)
    print(f"Additional cost without battery: ${cost_difference_no_battery:.2f}")
    print(f"Additional cost with battery: ${cost_difference_with_battery:.2f}")
    print("-"*50)
    print(f"Cooler day savings from battery: ${base_savings:.2f}")
    print(f"Warmer day savings from battery: ${hot_savings:.2f}")
    print(f"Additional savings due to temperature: ${additional_savings:.2f}")
    print("="*50)
    print("RECOMMENDATION:")
    print(f"A temperature increase of {temp_difference:.1f}°C will increase your energy consumption by {consumption_difference:.2f} kWh per dwelling")
    print(f"This will cost an additional ${cost_difference_no_battery:.2f} per dwelling without battery optimization.")
    print(f"Using a 60kWh battery during peak hours will save you ${hot_savings:.2f} per day during warmer temperatures.")
    print(f"The battery provides an additional ${additional_savings:.2f} in savings on warmer days compared to cooler days.")
    print("="*50)
    
    # Create comparative plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15))
    
    # Plot 1: Temperature comparison
    ax1.plot(range(24), base_results['hourly_temps'], 'b-o', label=f'Cooler Day ({base_date})')
    ax1.plot(range(24), hot_results['hourly_temps'], 'r-o', label=f'Warmer Day ({hot_date})')
    ax1.set_title('Temperature Comparison')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Consumption comparison
    ax2.plot(range(24), base_results['baseline_consumption'], 'b-', label=f'Cooler Day Consumption')
    ax2.plot(range(24), hot_results['baseline_consumption'], 'r-', label=f'Warmer Day Consumption')
    ax2.set_title('Energy Consumption Comparison')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Energy (kWh per dwelling)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Battery savings comparison
    base_savings_hourly = [b-o for b, o in zip(base_results['baseline_consumption'], base_results['optimized_consumption'])]
    hot_savings_hourly = [b-o for b, o in zip(hot_results['baseline_consumption'], hot_results['optimized_consumption'])]
    
    ax3.bar(np.array(range(24))-0.2, base_savings_hourly, width=0.4, color='blue', alpha=0.7, label=f'Cooler Day Savings')
    ax3.bar(np.array(range(24))+0.2, hot_savings_hourly, width=0.4, color='red', alpha=0.7, label=f'Warmer Day Savings')
    ax3.set_title('Battery Savings Comparison')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Energy Savings (kWh)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return comparison


def analyze_monthly_potential(weather_df, energy_df, optimizer, month, year=2018, num_dwellings=449880):
    """
    Analyze monthly potential savings from battery optimization.
    
    Parameters:
    weather_df (DataFrame): Cleaned weather data
    energy_df (DataFrame): Cleaned energy data
    optimizer (BatteryOptimizer): Configured optimizer object
    month (int): Month to analyze (1-12)
    year (int): Year to analyze
    num_dwellings (int): Number of dwellings for scaling
    
    Returns:
    DataFrame: Daily savings potential for the month
    """
    # Filter data for the specified month
    start_date = pd.to_datetime(f'{year}-{month:02d}-01')
    if month == 12:
        end_date = pd.to_datetime(f'{year+1}-01-01')
    else:
        end_date = pd.to_datetime(f'{year}-{month+1:02d}-01')
    
    month_energy = energy_df[(energy_df['datetime'] >= start_date) & 
                            (energy_df['datetime'] < end_date)]
    
    if month_energy.empty:
        print(f"No energy data available for {year}-{month:02d}")
        return None
    
    # Group by day and calculate daily savings
    daily_savings = []
    for date, day_data in month_energy.groupby(month_energy['datetime'].dt.date):
        # Resample to hourly data
        hourly_energy = day_data.set_index('datetime').resample('h').mean()
        hourly_consumption = hourly_energy['load_per_dwelling'].values
        
        # If we have missing hours, fill them using linear interpolation
        if len(hourly_consumption) < 24:
            complete_index = pd.date_range(start=date, end=date + timedelta(days=1), freq='h', closed='left')
            complete_series = pd.Series(index=complete_index)
            for idx, value in zip(hourly_energy.index, hourly_consumption):
                complete_series[idx] = value
            complete_series = complete_series.interpolate(method='linear')
            hourly_consumption = complete_series.values
        
        # Run optimization
        results = optimizer.optimize_battery_usage(hourly_consumption)
        
        daily_savings.append({
            'date': date,
            'consumption': sum(hourly_consumption),
            'savings_per_dwelling': results['savings'],
            'total_savings': results['savings'] * num_dwellings
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(daily_savings)
    
    # Plot monthly results
    if not results_df.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Get month name
        month_name = pd.to_datetime(f'{year}-{month:02d}-01').strftime('%B %Y')
        
        # Plot consumption
        ax1.plot(pd.to_datetime(results_df['date']), results_df['consumption'], 'b-o')
        ax1.set_title(f'Daily Energy Consumption - {month_name}')
        ax1.set_ylabel('Energy (kWh per dwelling)')
        ax1.grid(True)
        
        # Format x-axis to show day of month
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        
        # Plot savings
        ax2.bar(pd.to_datetime(results_df['date']), results_df['savings_per_dwelling'], color='g')
        ax2.set_title(f'Daily Savings per Dwelling - {month_name}')
        ax2.set_xlabel('Day of Month')
        ax2.set_ylabel('Savings ($)')
        ax2.grid(True)
        
        # Format x-axis to show day of month
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nMONTHLY BATTERY SAVINGS POTENTIAL - {month_name}")
        print("="*50)
        print(f"Average daily consumption: {results_df['consumption'].mean():.2f} kWh per dwelling")
        print(f"Average daily savings: ${results_df['savings_per_dwelling'].mean():.2f} per dwelling")
        print(f"Monthly savings per dwelling: ${results_df['savings_per_dwelling'].sum():.2f}")
        print(f"Total monthly savings for all dwellings: ${results_df['total_savings'].sum():,.2f}")
        print("="*50)
    
    return results_df


# Main execution
if __name__ == "__main__":
    # Define the hourly rates for each city
    # Atlanta rates (existing)
    atlanta_rates_peak = [
        0.07, 0.07, 0.07, 0.07, 0.07, 0.07,  # 0-5
        0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07,  # 6-13
        0.24, 0.24, 0.24, 0.24, 0.24,  # 14-18
        0.24, 0.07, 0.07,  # 19-21
        0.07, 0.07  # 22-23
    ]

    atlanta_rates_offpeak = [
        0.07, 0.07, 0.07, 0.07, 0.07,  # 0-5
        0.07, 0.07, 0.07, 0.07, 0.07,
        0.07, 0.07, 0.07, 0.07, 0.07,
        0.07, 0.07, 0.07, 0.07, 0.07,
        0.07, 0.07, 0.07, 0.07
    ]

    # Ann Arbor rates (DTE Energy)
    ann_arbor_rates_summer = [
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08,  # 0-5 (off-peak)
        0.12, 0.12, 0.12, 0.12, 0.18, 0.18,  # 6-11 (mid-peak, peak)
        0.18, 0.18, 0.18, 0.18, 0.18, 0.18,  # 12-17 (peak)
        0.18, 0.12, 0.12, 0.12, 0.12, 0.08   # 18-23 (mid-peak, off-peak)
    ]

    ann_arbor_rates_winter = [
        0.07, 0.07, 0.07, 0.07, 0.07, 0.07,  # 0-5 (off-peak)
        0.11, 0.11, 0.11, 0.11, 0.15, 0.15,  # 6-11 (mid-peak, peak)
        0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  # 12-17 (peak)
        0.15, 0.11, 0.11, 0.11, 0.11, 0.07   # 18-23 (mid-peak, off-peak)
    ]

    # San Diego rates (SDG&E)
    sandiego_rates_summer = [
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  # 0-5 (off-peak)
        0.30, 0.30, 0.30, 0.30, 0.30, 0.30,  # 6-11 (peak)
        0.30, 0.30, 0.30, 0.30, 0.48, 0.48,  # 12-17 (peak, super-peak)
        0.48, 0.48, 0.48, 0.30, 0.30, 0.30   # 18-23 (super-peak, peak)
    ]

    sandiego_rates_winter = [
        0.22, 0.22, 0.22, 0.22, 0.22, 0.22,  # 0-5 (off-peak)
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  # 6-11 (peak)
        0.25, 0.25, 0.25, 0.25, 0.35, 0.35,  # 12-17 (peak, super-peak)
        0.35, 0.35, 0.35, 0.25, 0.25, 0.25   # 18-23 (super-peak, peak)
    ]

    # Create optimizers for each city
    atlanta_peak_optimizer = BatteryOptimizer(hourly_rates=atlanta_rates_peak, battery_capacity=60)
    atlanta_offpeak_optimizer = BatteryOptimizer(hourly_rates=atlanta_rates_offpeak, battery_capacity=60)
    
    ann_arbor_summer_optimizer = BatteryOptimizer(hourly_rates=ann_arbor_rates_summer, battery_capacity=60)
    ann_arbor_winter_optimizer = BatteryOptimizer(hourly_rates=ann_arbor_rates_winter, battery_capacity=60)
    
    sandiego_summer_optimizer = BatteryOptimizer(hourly_rates=sandiego_rates_summer, battery_capacity=60)
    sandiego_winter_optimizer = BatteryOptimizer(hourly_rates=sandiego_rates_winter, battery_capacity=60)

    # Analyze each city
    for city, city_data in CITY_DATA.items():
        print(f"\nAnalyzing {city}...")
        
        # Load and clean data
        print("Loading and processing data files...")
        
        # Load weather data
        weather_data = pd.read_csv(city_data['weather_file'], low_memory=False)
        weather_df = clean_weather_data(weather_data)
        
        # Load energy data
        energy_data = pd.read_csv(city_data['energy_file'])
        energy_df = process_energy_data(energy_data, city_data['dwellings'])
        
        # Process the data to hourly format for analysis
        energy_hourly = energy_df.set_index('datetime').resample('h').mean()
        weather_hourly = weather_df.set_index('datetime').resample('h').mean()
        
        # Merge datasets
        combined_df = pd.merge(
            energy_hourly,
            weather_hourly,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Reset index
        combined_df = combined_df.reset_index()
        
        # Filter dataframes to include only valid dates
        valid_dates = set(combined_df['datetime'].dt.date)
        weather_df_filtered = weather_df[weather_df['datetime'].dt.date.isin(valid_dates)]
        energy_df_filtered = energy_df[energy_df['datetime'].dt.date.isin(valid_dates)]
        
        # Find summer and winter dates
        summer_dates = combined_df[(combined_df['datetime'].dt.month >= 6) & 
                                 (combined_df['datetime'].dt.month <= 8)]['datetime'].dt.date.unique()
        winter_dates = combined_df[(combined_df['datetime'].dt.month == 12) | 
                                 (combined_df['datetime'].dt.month <= 2)]['datetime'].dt.date.unique()
        
        # Select representative dates
        summer_date = str(summer_dates[len(summer_dates)//2])
        winter_date = str(winter_dates[len(winter_dates)//2])
        
        print(f"Selected representative summer date: {summer_date}")
        print(f"Selected representative winter date: {winter_date}")
        
        # Select appropriate optimizer based on city
        if city == 'Atlanta':
            summer_optimizer = atlanta_peak_optimizer
            winter_optimizer = atlanta_offpeak_optimizer
        elif city == 'Ann Arbor':
            summer_optimizer = ann_arbor_summer_optimizer
            winter_optimizer = ann_arbor_winter_optimizer
        else:  # San Diego
            summer_optimizer = sandiego_summer_optimizer
            winter_optimizer = sandiego_winter_optimizer
        
        # Analyze specific days
        print("\nAnalyzing the selected summer day")
        summer_day = analyze_day(weather_df_filtered, energy_df_filtered, summer_date, summer_optimizer)
        
        print("\nAnalyzing the selected winter day")
        winter_day = analyze_day(weather_df_filtered, energy_df_filtered, winter_date, winter_optimizer)
        
        # Compare summer vs winter day
        print("\nComparing summer day vs winter day")
        seasonal_comparison = compare_temperature_impact(weather_df_filtered, energy_df_filtered, 
                                                      winter_date, summer_date, summer_optimizer, winter_optimizer)
        
        # Analyze monthly potential
        summer_month = pd.to_datetime(summer_date).month
        winter_month = pd.to_datetime(winter_date).month
        
        print(f"\nAnalyzing potential savings for month {summer_month} (summer)")
        summer_analysis = analyze_monthly_potential(weather_df_filtered, energy_df_filtered, summer_optimizer, 
                                                 month=summer_month, num_dwellings=city_data['dwellings'])
        
        print(f"\nAnalyzing potential savings for month {winter_month} (winter)")
        winter_analysis = analyze_monthly_potential(weather_df_filtered, energy_df_filtered, winter_optimizer, 
                                                 month=winter_month, num_dwellings=city_data['dwellings'])
        
        # Calculate and display annual savings estimate
        if summer_analysis is not None and winter_analysis is not None:
            summer_daily_avg = summer_analysis['savings_per_dwelling'].mean()
            winter_daily_avg = winter_analysis['savings_per_dwelling'].mean()
            
            # Calculate annual savings based on city's rate structure
            if city == 'Atlanta':
                # Atlanta has peak rates only in summer
                annual_estimate = summer_daily_avg * 122  # Summer months
            elif city == 'Ann Arbor':
                # Ann Arbor has different rates in summer and winter
                annual_estimate = (summer_daily_avg * 122) + (winter_daily_avg * 243)
            else:  # San Diego
                # San Diego has different rates in summer and winter
                annual_estimate = (summer_daily_avg * 153) + (winter_daily_avg * 212)
            
            print("\nANNUAL SAVINGS ESTIMATE")
            print("="*50)
            print(f"Estimated average daily savings in summer: ${summer_daily_avg:.2f} per dwelling")
            print(f"Estimated average daily savings in winter: ${winter_daily_avg:.2f} per dwelling")
            print(f"Estimated annual savings: ${annual_estimate:.2f} per dwelling")
            print(f"Total annual savings for all dwellings: ${annual_estimate * city_data['dwellings']:,.2f}")
            print(f"Payback period for 60kWh battery (assuming $1,000 cost): {1000/annual_estimate:.1f} years")
            print("="*50)