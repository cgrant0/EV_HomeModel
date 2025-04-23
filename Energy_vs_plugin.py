import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Define the city data with dwelling counts, file paths, and corresponding county names
CITY_DATA = {
    'Atlanta': {
        'dwellings': 449880,
        'energy_file': 'Atlanta/Atlanta_15_minute_timeseries_data.csv',
        'county': 'Fulton'
    },
    'Ann Arbor': {
        'dwellings': 149394,
        'energy_file': 'Ann_Arbor/AnnArbor_15_minute_timeseries_data_year.csv',
        'county': 'Washtenaw'
    },
    'San Diego': {
        'dwellings': 1187651,
        'energy_file': 'San_Diego/SanDiego_15_minute_timeseries_data.csv',
        'county': 'San Diego'
    }
}

# Extract data points for EV charging at home from the graph (red bars)
# Hours are in 24-hour format and values are percentages of time spent charging
ev_charging_home_data = {
    0: 4.7,   # 12:00 AM
    1: 4.0,   # 1:00 AM
    2: 3.5,   # 2:00 AM
    3: 2.8,   # 3:00 AM
    4: 2.0,   # 4:00 AM
    5: 1.2,   # 5:00 AM
    6: 1.0,   # 6:00 AM
    7: 1.2,   # 7:00 AM
    8: 1.4,   # 8:00 AM
    9: 1.3,   # 9:00 AM
    10: 1.5,  # 10:00 AM
    11: 2.0,  # 11:00 AM
    12: 2.7,  # 12:00 PM
    13: 3.4,  # 1:00 PM
    14: 3.7,  # 2:00 PM
    15: 4.0,  # 3:00 PM
    16: 5.0,  # 4:00 PM
    17: 6.0,  # 5:00 PM
    18: 7.3,  # 6:00 PM
    19: 8.2,  # 7:00 PM
    20: 8.1,  # 8:00 PM
    21: 6.5,  # 9:00 PM
    22: 6.0,  # 10:00 PM
    23: 5.0,  # 11:00 PM
}

# Convert to dataframe for easier plotting
ev_df = pd.DataFrame({
    'Hour': list(ev_charging_home_data.keys()), 
    'Charging_Percentage': list(ev_charging_home_data.values())
})

# Function to load and process power outage data
def load_power_outage_data(year=2023):
    """Load and process outage and customer count data"""
    try:
        outages_df = pd.read_csv('Power_Data/outages_{year}.csv')
        customers_df = pd.read_csv('Power_Data/county_FIPS_customers.csv')
        
        # Convert run_start_time to datetime
        outages_df['run_start_time'] = pd.to_datetime(outages_df['run_start_time'])
        
        # Extract hour from timestamp
        outages_df['hour'] = outages_df['run_start_time'].dt.hour
        
        # Add date column for filtering
        outages_df['date'] = outages_df['run_start_time'].dt.date
        
        return outages_df, customers_df
    except FileNotFoundError as e:
        print(f"Error loading power outage data: {e}")
        return None, None

# Function to get power outage data for a specific county on a specific date
def get_power_outage_data(county_name, specific_date, outages_df, customers_df):
    """Get power outage data for a specific county and date"""
    if outages_df is None or customers_df is None:
        print("Power outage data not available")
        return None
    
    # Filter outages data for the specified county
    county_outages = outages_df[outages_df['county'] == county_name]
    
    if county_outages.empty:
        print(f"No outage data found for {county_name} County")
        return None
    
    # Get the FIPS code for this county
    fips_code = str(county_outages['fips_code'].iloc[0])
    
    # Get total customer count for this county
    county_customers = customers_df[customers_df['County_FIPS'] == fips_code]
    
    if county_customers.empty:
        print(f"No customer data found for {county_name} County (FIPS: {fips_code})")
        return None
    
    total_customers = county_customers['Customers'].iloc[0]
    
    # Filter by specific date
    date_mask = county_outages['date'] == specific_date
    filtered_outages = county_outages[date_mask]
    
    if filtered_outages.empty:
        print(f"No outage data for {county_name} County on {specific_date}")
        # If no data for specific date, fall back to average across all dates
        print(f"Using average outage data for {county_name} County instead")
        # Group by hour and calculate average outages
        hourly_avg = county_outages.groupby('hour')['sum'].mean()
    else:
        # Group by hour and calculate sum of outages for the specific date
        hourly_avg = filtered_outages.groupby('hour')['sum'].mean()
    
    # Calculate percentage of customers affected
    hourly_percentages = (hourly_avg / total_customers) * 100
    
    # Convert to DataFrame
    outage_df = pd.DataFrame({
        'Hour': hourly_percentages.index,
        'Outage_Percentage': hourly_percentages.values
    })
    
    # Ensure we have data for all 24 hours
    hours_df = pd.DataFrame({'Hour': range(24)})
    outage_df = pd.merge(hours_df, outage_df, on='Hour', how='left').fillna(0)
    
    return outage_df

# Function to analyze city's peak day
def analyze_city_peak_day(city_name, city_info):
    print(f"Processing {city_name}...")
    
    # Load the data
    try:
        df = pd.read_csv(city_info['energy_file'])
        print(f"Data loaded for {city_name} with {len(df)} rows")
    except FileNotFoundError:
        print(f"File not found: {city_info['energy_file']} - Check your directory structure")
        return None
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp (EST)'])
    
    # Normalize total electricity by number of dwellings
    total_column = 'baseline.out.electricity.total.energy_consumption.kwh'
    df[total_column] = df[total_column] / city_info['dwellings']
    
    # Add date column and hour column
    df['Date'] = df['Timestamp'].dt.date
    df['Hour'] = df['Timestamp'].dt.hour
    
    # Group by date and sum to find the day with highest total consumption
    daily_totals = df.groupby('Date')[total_column].sum().reset_index()
    peak_date = daily_totals.loc[daily_totals[total_column].idxmax()]['Date']
    
    print(f"Peak date for {city_name}: {peak_date}")
    
    # Filter data for the peak day
    peak_day_data = df[df['Date'] == peak_date].copy()
    
    # Group by hour for the peak day
    hourly_data = peak_day_data.groupby('Hour')[total_column].sum().reset_index()
    hourly_data.columns = ['Hour', 'Total_Energy']
    
    return peak_date, hourly_data

# Main plotting function
def create_triple_overlay_plots():
    # Load power outage data
    outages_df, customers_df = load_power_outage_data()
    power_outage_available = outages_df is not None and customers_df is not None
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 20))
    fig.suptitle('Energy Usage, EV Charging, and Power Outages Comparison', fontsize=18, y=0.92)
    
    # Process each city
    for i, (city_name, city_info) in enumerate(CITY_DATA.items()):
        result = analyze_city_peak_day(city_name, city_info)
        
        if result is None:
            print(f"Skipping {city_name} due to data loading error")
            # Create a blank plot with an error message
            axes[i].text(0.5, 0.5, f"Could not load electricity data for {city_name}", 
                        horizontalalignment='center', verticalalignment='center')
            axes[i].set_title(f"{city_name} - Data not available")
            continue
        
        peak_date, hourly_data = result
        
        # Plot electricity usage (primary axis)
        ax1 = axes[i]
        color1 = 'tab:blue'
        line1 = ax1.plot(hourly_data['Hour'], hourly_data['Total_Energy'], color=color1, 
                 marker='o', linewidth=2, markersize=6, label='Total Electricity')
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Electricity Consumption\n(kWh per dwelling)', color=color1, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xlim(0, 23)
        ax1.grid(True, alpha=0.3)
        
        # Create first secondary y-axis for EV charging
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        line2 = ax2.plot(ev_df['Hour'], ev_df['Charging_Percentage'], color=color2, 
                 marker='s', linestyle='--', linewidth=2, markersize=6, label='EV Charging at Home')
        ax2.set_ylabel('EV Charging at Home (%)', color=color2, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 12)  # Setting limit slightly higher than max value
        
        # If power outage data is available, add it as a third dataset
        if power_outage_available:
            # Create second secondary y-axis for power outages
            ax3 = ax1.twinx()
            # Offset the position to the right
            ax3.spines['right'].set_position(('outward', 70))
            color3 = 'tab:green'
            
            # Get power outage data for the county on the peak date
            county_name = city_info['county']
            outage_data = get_power_outage_data(county_name, peak_date, outages_df, customers_df)
            
            if outage_data is not None:
                line3 = ax3.plot(outage_data['Hour'], outage_data['Outage_Percentage'], color=color3, 
                         marker='^', linestyle='-.', linewidth=2, markersize=6, label='Power Outages')
                ax3.set_ylabel('Customers Affected by\nPower Outages (%)', color=color3, fontsize=12)
                ax3.tick_params(axis='y', labelcolor=color3)
                
                # Set y-limit for outages based on data, with a minimum of 1%
                max_outage = max(0.1, outage_data['Outage_Percentage'].max() * 1.2)
                ax3.set_ylim(0, max_outage)
                
                # Combine legends from all three axes
                lines = line1 + line2 + line3
                labels = ['Total Electricity', 'EV Charging at Home', 'Power Outages']
            else:
                # Only combine legends from two axes if outage data is not available
                lines = line1 + line2
                labels = ['Total Electricity', 'EV Charging at Home']
        else:
            # Only combine legends from two axes if power outage data is not available at all
            lines = line1 + line2
            labels = ['Total Electricity', 'EV Charging at Home']
        
        # Add legend
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        
        # Set title
        ax1.set_title(f"{city_name} - Peak Electricity Usage Day ({peak_date})", fontsize=14)
        
        # Set x-ticks every 2 hours
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjust for the suptitle
    
    # Save the figure
    plt.savefig('energy_ev_outages_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Analysis complete! Figure saved as 'energy_ev_outages_comparison.png'")
    return fig

def create_combined_energy_comparison():
    """
    Creates a single graph overlaying the energy usage patterns from all three cities
    along with the EV charging data displayed as a bar graph.
    """
    plt.figure(figsize=(14, 8))
    
    # Create a color palette for the cities
    city_colors = {
        'Atlanta': 'tab:blue',
        'Ann Arbor': 'tab:purple',
        'San Diego': 'tab:green'
    }
    
    # Get primary axis
    ax1 = plt.gca()
    
    # Process each city to get the peak day data
    legend_elements = []
    
    for city_name, city_info in CITY_DATA.items():
        result = analyze_city_peak_day(city_name, city_info)
        
        if result is None:
            print(f"Skipping {city_name} due to data loading error")
            continue
        
        peak_date, hourly_data = result
        
        # Plot electricity usage for this city
        line = ax1.plot(hourly_data['Hour'], hourly_data['Total_Energy'], 
                 color=city_colors[city_name], marker='o', linewidth=2, 
                 markersize=6, label=f"{city_name} ({peak_date})")
        
        legend_elements.append(line[0])
    
    # Add EV charging data on a secondary axis as bars
    ax2 = ax1.twinx()
    ev_color = 'tab:red'
    
    # Create bar chart for EV data with slight transparency
    bars = ax2.bar(ev_df['Hour'], ev_df['Charging_Percentage'], 
            color=ev_color, alpha=0.25, width=0.7, 
            label='EV Charging at Home (%)')
    
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    
    # Add a proxy artist for the legend
    from matplotlib.patches import Patch
    legend_elements.append(Patch(facecolor=ev_color, alpha=0.5, label='EV Charging at Home (%)'))
    
    # Set labels and formatting
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Electricity Consumption (kWh per dwelling)', fontsize=12)
    ax2.set_ylabel('EV Charging at Home (%)', color=ev_color, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=ev_color)
    
    # Set x-axis ticks every 2 hours
    ax1.set_xlim(-0.5, 23.5)  # Adjust limits to show full bars
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    
    # Add grid (behind the bars)
    ax1.grid(True, alpha=0.3, zorder=0)
    
    # Add legend with all elements
    labels = [element.get_label() for element in legend_elements]
    plt.legend(legend_elements, labels, loc='upper left', fontsize=10)
    
    # Set title
    plt.title('Comparison of Peak Day Energy Usage Across Cities with EV Charging Pattern', fontsize=16)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('combined_energy_comparison_bar.png', dpi=300, bbox_inches='tight')
    
    print("Combined analysis complete! Figure saved as 'combined_energy_comparison_bar.png'")
    return plt.gcf()

def find_max_hourly_consumption():
    """
    Finds the highest kWh electricity consumption for a single hour by a single dwelling
    in each target city.
    
    Returns:
        dict: A dictionary with city names as keys and tuples of (max_consumption, date, hour) as values
    """
    results = {}
    
    for city_name, city_info in CITY_DATA.items():
        print(f"Finding max hourly consumption for {city_name}...")
        
        # Load the data
        try:
            df = pd.read_csv(city_info['energy_file'])
            print(f"Data loaded for {city_name} with {len(df)} rows")
        except FileNotFoundError:
            print(f"File not found: {city_info['energy_file']} - Check your directory structure")
            results[city_name] = None
            continue
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp (EST)'])
        
        # Normalize total electricity by number of dwellings
        total_column = 'baseline.out.electricity.total.energy_consumption.kwh'
        df[total_column] = df[total_column] / city_info['dwellings']
        
        # Add date column and hour column
        df['Date'] = df['Timestamp'].dt.date
        df['Hour'] = df['Timestamp'].dt.hour
        
        # Find the maximum consumption value
        max_consumption = df[total_column].max()
        
        # Find the row with the maximum consumption
        max_row = df.loc[df[total_column].idxmax()]
        
        # Extract date and hour
        max_date = max_row['Date']
        max_hour = max_row['Hour']
        
        # Store the results
        results[city_name] = (max_consumption, max_date, max_hour)
        
        print(f"Max consumption for {city_name}: {max_consumption:.4f} kWh per dwelling")
        print(f"Occurred on {max_date} at hour {max_hour}")
    
    return results

if __name__ == "__main__":
    #create_triple_overlay_plots()
    #create_combined_energy_comparison()
    
    # Find and display max hourly consumption for each city
    max_consumption_results = find_max_hourly_consumption()
    
    # Print a summary table
    print("\nMaximum Hourly Electricity Consumption by City:")
    print("-" * 80)
    print(f"{'City':<15} {'Max Consumption (kWh)':<20} {'Date':<15} {'Hour':<10}")
    print("-" * 80)
    
    for city, result in max_consumption_results.items():
        if result is not None:
            max_consumption, max_date, max_hour = result
            print(f"{city:<15} {max_consumption:<20.4f} {max_date} {max_hour}")
        else:
            print(f"{city:<15} {'Data not available':<20} {'N/A':<15} {'N/A':<10}")
    
    print("-" * 80)