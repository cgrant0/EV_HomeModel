import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Hours are in 24-hour format and values are percentages of time spent charging
# Manually extracted from Slide 22, Presentation 4
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

def load_and_process_data():
    """Load and process outage and customer count data"""
    # Load the datasets
    outages_df = pd.read_csv('Power_Data/outages_2023.csv')
    customers_df = pd.read_csv('Power_Data/county_FIPS_customers.csv')
    
    # Convert run_start_time to datetime
    outages_df['run_start_time'] = pd.to_datetime(outages_df['run_start_time'])
    
    # Extract hour from timestamp
    outages_df['hour'] = outages_df['run_start_time'].dt.hour
    
    # Add date column for filtering
    outages_df['date'] = outages_df['run_start_time'].dt.date
    
    return outages_df, customers_df

def get_county_data(outages_df, customers_df, county_name):
    """Extract data for a specific county"""
    # Filter outages data for the specified county
    county_outages = outages_df[outages_df['county'] == county_name]
    
    if county_outages.empty:
        print(f"No data found for {county_name} County")
        return None, None
    
    # Get the FIPS code for this county
    fips_code = str(county_outages['fips_code'].iloc[0])
    
    # Get total customer count for this county
    county_customers = customers_df[customers_df['County_FIPS'] == fips_code]
    
    if county_customers.empty:
        print(f"No customer data found for {county_name} County (FIPS: {fips_code})")
        return None, None
    
    total_customers = county_customers['Customers'].iloc[0]
    
    return county_outages, total_customers

def calculate_hourly_data(outages_df, total_customers, start_date=None, end_date=None, asPercentage=True):
    """Calculate percentage of customers affected by hour"""
    # Filter by date range if specified
    if start_date and end_date:
        mask = (outages_df['date'] >= start_date) & (outages_df['date'] <= end_date)
        filtered_outages = outages_df[mask]
    else:
        filtered_outages = outages_df
    
    if filtered_outages.empty:
        print("No data available for the specified date range")
        return None
    
    # Group by hour and calculate average outages
    hourly_avg = filtered_outages.groupby('hour')['sum'].mean()
    
    # Calculate percentage of customers affected
    if asPercentage:
        hourly_data = (hourly_avg / total_customers) * 100
    else:
        hourly_data = round(hourly_avg)
    
    return hourly_data

def plot_hourly_outages(county_names, start_date=None, end_date=None, asPercentage=True):
    """Generate plots for specified counties"""
    outages_df, customers_df = load_and_process_data()
    
    # Create a figure with increased spacing between subplots
    fig, axes = plt.subplots(len(county_names), 1, figsize=(12, 6*len(county_names)))
    
    # If only one county, make axes iterable
    if len(county_names) == 1:
        axes = [axes]
    
    # Date range description for the title
    date_range_str = f"({start_date} to {end_date})" if start_date and end_date else "(Full Year 2023)"
    
    for i, county_name in enumerate(county_names):
        county_outages, total_customers = get_county_data(outages_df, customers_df, county_name)
        
        if county_outages is None or total_customers is None:
            axes[i].text(0.5, 0.5, f"No data available for {county_name} County", 
                        horizontalalignment='center', fontsize=12)
            axes[i].set_title(f"{county_name} County - No Data")
            continue
            
        hourly_percentages = calculate_hourly_data(county_outages, total_customers, start_date, end_date, asPercentage)
        
        if hourly_percentages is None:
            axes[i].text(0.5, 0.5, f"No data available for {county_name} County in the specified date range", 
                        horizontalalignment='center', fontsize=12)
            axes[i].set_title(f"{county_name} County - No Data")
            continue
        
        # Plot the data
        bars = sns.barplot(x=hourly_percentages.index, y=hourly_percentages.values, ax=axes[i], hue=hourly_percentages.values, palette='autumn', legend=False)
        
        # Set titles and labels
        axes[i].set_title(f"{county_name} County Power Outages by Hour of Day {date_range_str}", fontsize=14)
        axes[i].set_xlabel('Hour of Day', fontsize=12)
        axes[i].set_ylabel('Average Number of Customers Affected', fontsize=12)
        axes[i].set_xticks(range(24))
        axes[i].set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
        
        # Get the current y-limit
        y_max = axes[i].get_ylim()[1]
        # Set a small buffer (5% of the max value) for the label placement
        buffer = y_max * 0.05
        
        # Add value labels directly above each bar with proper positioning
        for j, bar in enumerate(bars.patches):
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                # Position the text at the top of the bar
                axes[i].text(
                    bar.get_x() + bar.get_width()/2., 
                    height + buffer,
                    f'{height:.0f}', 
                    ha="center", 
                    va="bottom",
                    fontsize=9,
                    fontweight='bold'
                )
        
        # Adjust y-axis to make room for labels
        axes[i].set_ylim(0, y_max * 1.15)
    
    # Add more space between subplots
    plt.subplots_adjust(hspace=0.4)
    
    # Make the layout tight but with enough space
    plt.tight_layout()
    
    # Save the figure
    save_name = f"power_outages_{'_'.join(county_names)}_{start_date}_{end_date}.png" if start_date and end_date else f"power_outages_{'_'.join(county_names)}_full_year.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    
    return fig

def plot_historical_trends(county_names):
    """
    Generate a single plot showing the historical trend of power outages from 2014-2023
    for the specified counties, measuring absolute customer counts rather than percentages.
    
    Parameters:
    county_names (list): List of county names to analyze
    
    Returns:
    fig: The matplotlib figure object
    """
    # Load customer data for reference
    customers_df = pd.read_csv('Power_Data/county_FIPS_customers.csv')
    
    # Dictionary to store yearly average outage data by county
    yearly_data = {county: {} for county in county_names}
    
    # Dictionary to store FIPS codes by county name
    county_fips = {}
    
    # Process data for each year from 2014 to 2023
    years = range(2014, 2023)
    for year in years:
        # Construct filename
        filename = f"Power_Data/outages_{year}.csv"
        
        # Check if file exists
        #if not os.path.exists(filename):
         #   print(f"Warning: {filename} does not exist. Skipping year {year}.")
          #  continue
        
        print(f"Processing data for year {year}...")
        
        # Load data for this year
        year_df = pd.read_csv(filename)
        
        # Process data for each county
        for county_name in county_names:
            # Filter data for this county
            county_data = year_df[year_df['county'] == county_name]
            
            if county_data.empty:
                print(f"No data found for {county_name} County in {year}")
                yearly_data[county_name][year] = 0  # Set to 0 if no data
                continue
            
            # Store FIPS code if we don't have it yet
            if county_name not in county_fips and not county_data.empty:
                county_fips[county_name] = county_data['fips_code'].iloc[0]
                
            # Calculate the average number of customers affected
            # First convert timestamps if they exist
            if 'run_start_time' in county_data.columns:
                county_data['run_start_time'] = pd.to_datetime(county_data['run_start_time'])
            
            # Calculate average number of customers affected
            if year == 2023:
                avg_customers_affected = county_data['sum'].mean()
            else:
                avg_customers_affected = county_data['customers_out'].mean()

            yearly_data[county_name][year] = avg_customers_affected
    
    # Create a DataFrame from the collected data
    trend_data = []
    for county, yearly_values in yearly_data.items():
        for year, value in yearly_values.items():
            trend_data.append({
                'County': county,
                'Year': year,
                'Average Customers Affected': value
            })
    
    trend_df = pd.DataFrame(trend_data)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Define a color palette for the counties
    colors = sns.color_palette("husl", len(county_names))
    
    # Plot each county's trend line
    for i, county in enumerate(county_names):
        county_data = trend_df[trend_df['County'] == county]
        if not county_data.empty:
            plt.plot(
                county_data['Year'], 
                county_data['Average Customers Affected'],
                marker='o',
                linewidth=2.5,
                markersize=8,
                label=f"{county} County",
                color=colors[i]
            )
    
    # Add labels and title
    plt.title('Historical Trend of Power Outages (2014-2023)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Number of Customers Affected', fontsize=14)
    plt.xticks(years, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Ensure all data points are visible
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"historical_power_outages_{'_'.join(county_names)}_2014_2023.png", dpi=300, bbox_inches='tight')
    
    print(f"Historical trend analysis for {', '.join(county_names)} counties completed.")
    return plt.gcf()

def create_ev_outage_comparison():
    """
    Creates a single graph overlaying the EV charging patterns with power outage trends for each county.
    Uses power outage data averaged over the full year of 2023.
    Shows total number of customers affected rather than percentages.
    """
    # Load outage data
    outages_df, customers_df = load_and_process_data()
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Set up colors for counties
    county_colors = {
        'Fulton': 'tab:blue',
        'Washtenaw': 'tab:purple',
        'San Diego': 'tab:green'
    }
    
    # Primary y-axis for EV charging
    ax1 = plt.gca()
    ev_color = 'tab:red'
    
    # Plot EV charging data on primary axis
    ev_line = ax1.plot(ev_df['Hour'], ev_df['Charging_Percentage'], 
               color=ev_color, marker='s', linewidth=3, 
               markersize=8, label='EV Charging at Home (%)')
    
    # Set up primary axis labels
    ax1.set_xlabel('Hour of Day', fontsize=14)
    ax1.set_ylabel('EV Charging at Home (%)', color=ev_color, fontsize=14)
    ax1.tick_params(axis='y', labelcolor=ev_color)
    ax1.set_ylim(0, 10)  # Adjusted to match EV charging data range
    
    # Create secondary y-axis for outage data
    ax2 = ax1.twinx()
    
    # List to store legend elements
    legend_elements = ev_line.copy()
    
    # Plot outage data for each county
    county_names = ["Fulton", "Washtenaw", "San Diego"]
    for county_name in county_names:
        # Get county data
        county_outages, total_customers = get_county_data(outages_df, customers_df, county_name)
        
        if county_outages is None or total_customers is None:
            print(f"Skipping {county_name} County due to missing data")
            continue
            
        # Calculate hourly TOTAL customers affected (not percentage)
        # Set asPercentage=False to get absolute numbers
        hourly_customers = calculate_hourly_data(county_outages, total_customers, asPercentage=False)
        
        if hourly_customers is None:
            print(f"No hourly data available for {county_name} County")
            continue
        
        # Ensure we have data for all 24 hours
        full_hours = pd.Series(index=range(24), data=0.0)
        for hour, value in hourly_customers.items():
            if hour in range(24):
                full_hours[hour] = value
        
        # Plot county outage data
        line = ax2.plot(full_hours.index, full_hours.values, 
                     color=county_colors[county_name], marker='o', 
                     linestyle='--', linewidth=2, markersize=6, 
                     label=f"{county_name} County Outages")
        
        legend_elements.extend(line)
    
    # Set up secondary axis labels
    ax2.set_ylabel('Total Customers Affected by Power Outages', fontsize=14)
    
    # Ensure y-axis starts at 0
    ax2.set_ylim(bottom=0)
    
    # Set x-axis ticks for hours
    ax1.set_xlim(0, 23)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Format y-axis for large numbers
    from matplotlib.ticker import FuncFormatter
    def format_thousands(x, pos):
        return f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'
        
    ax2.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    
    # Combine all legend elements
    labels = [line.get_label() for line in legend_elements]
    plt.legend(legend_elements, labels, loc='upper left', fontsize=12)
    
    # Add title
    plt.title('EV Charging Patterns vs. Total Customers Affected by Outages (2023)', fontsize=18)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('ev_charging_vs_total_outages.png', dpi=300, bbox_inches='tight')
    
    print("EV charging vs total customers affected by outages comparison complete! Figure saved as 'ev_charging_vs_total_outages.png'")
    return plt.gcf()

def analyze_outage_hours(county_names):
    """
    Analyze and plot the total hours of power outages per year and workday hours (8am-6pm)
    for specified counties, including average outage hours per home.
    """
    # Load customer data for reference
    customers_df = pd.read_csv('Power_Data/county_FIPS_customers.csv')
    
    # Dictionary to store yearly outage hours data by county
    yearly_data = {county: {
        'total_hours': {}, 
        'workday_hours': {},
        'avg_hours_per_home': {},
        'avg_workday_hours_per_home': {},
        'total_customer_hours': {}
    } for county in county_names}
    
    # Process data for each year from 2014 to 2023
    years = range(2014, 2024)
    for year in years:
        # Construct filename
        filename = f"Power_Data/outages_{year}.csv"
        
        try:
            # Load data for this year
            year_df = pd.read_csv(filename)
            year_df['run_start_time'] = pd.to_datetime(year_df['run_start_time'])
            
            # Print column names for debugging
            if year == 2023:  # Only print for the most recent year
                print("\nAvailable columns in the data:")
                print(year_df.columns.tolist())
            
            # Process data for each county
            for county_name in county_names:
                # Filter data for this county
                county_data = year_df[year_df['county'] == county_name]
                
                if county_data.empty:
                    print(f"No data found for {county_name} County in {year}")
                    yearly_data[county_name]['total_hours'][year] = 0
                    yearly_data[county_name]['workday_hours'][year] = 0
                    yearly_data[county_name]['avg_hours_per_home'][year] = 0
                    yearly_data[county_name]['avg_workday_hours_per_home'][year] = 0
                    yearly_data[county_name]['total_customer_hours'][year] = 0
                    continue
                
                # Get total customers for this county
                fips_code = str(county_data['fips_code'].iloc[0])
                county_customers = customers_df[customers_df['County_FIPS'] == fips_code]
                if county_customers.empty:
                    print(f"No customer data found for {county_name} County (FIPS: {fips_code})")
                    continue
                total_customers = county_customers['Customers'].iloc[0]
                
                # Calculate total hours with outages
                total_hours = len(county_data)
                
                # Calculate workday hours (8am-6pm)
                workday_data = county_data[
                    (county_data['run_start_time'].dt.hour >= 8) & 
                    (county_data['run_start_time'].dt.hour < 18)
                ]
                workday_hours = len(workday_data)
                
                # Calculate total customer-hours of outage
                # Try different possible column names for customer count
                customer_count_column = None
                possible_columns = ['sum', 'customers_out', 'affected_customers', 'customer_count']
                
                for col in possible_columns:
                    if col in county_data.columns:
                        customer_count_column = col
                        break
                
                if customer_count_column is None:
                    print(f"Warning: Could not find customer count column in {year} data. Available columns: {county_data.columns.tolist()}")
                    total_customer_hours = 0
                    workday_customer_hours = 0
                else:
                    total_customer_hours = county_data[customer_count_column].sum()
                    workday_customer_hours = workday_data[customer_count_column].sum()
                
                # Calculate average hours per home
                avg_hours_per_home = total_customer_hours / total_customers if total_customers > 0 else 0
                avg_workday_hours_per_home = workday_customer_hours / total_customers if total_customers > 0 else 0
                
                yearly_data[county_name]['total_hours'][year] = total_hours
                yearly_data[county_name]['workday_hours'][year] = workday_hours
                yearly_data[county_name]['avg_hours_per_home'][year] = avg_hours_per_home
                yearly_data[county_name]['avg_workday_hours_per_home'][year] = avg_workday_hours_per_home
                yearly_data[county_name]['total_customer_hours'][year] = total_customer_hours
                
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Skipping year {year}.")
            continue
    
    # Print summary of outage hours for each county
    print("\nOutage Hours Summary:")
    print("=" * 120)
    for county in county_names:
        print(f"\n{county} County:")
        print("-" * 100)
        print("Year | Total Hours | Workday Hours | Avg Hours/Home | Avg Workday Hours/Home | Total Customer-Hours")
        print("-" * 100)
        for year in sorted(yearly_data[county]['total_hours'].keys()):
            total = yearly_data[county]['total_hours'][year]
            workday = yearly_data[county]['workday_hours'][year]
            avg_hours = yearly_data[county]['avg_hours_per_home'][year]
            avg_workday_hours = yearly_data[county]['avg_workday_hours_per_home'][year]
            customer_hours = yearly_data[county]['total_customer_hours'][year]
            print(f"{year} | {total:11d} | {workday:12d} | {avg_hours:13.2f} | {avg_workday_hours:20.2f} | {customer_hours:19.0f}")
    
    # Create the plots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 20))
    
    # Define a color palette for the counties
    colors = sns.color_palette("husl", len(county_names))
    
    # Plot total hours
    for i, county in enumerate(county_names):
        years_data = list(yearly_data[county]['total_hours'].keys())
        hours_data = list(yearly_data[county]['total_hours'].values())
        
        line = ax1.plot(years_data, hours_data, 
                marker='o', linewidth=2.5, markersize=8,
                label=f"{county} County", color=colors[i])[0]
        
        # Add data labels
        for x, y in zip(years_data, hours_data):
            ax1.annotate(f'{y}', 
                        (x, y),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9)
    
    ax1.set_title('Total Hours of Power Outages per Year', fontsize=16)
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Total Hours with Outages', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Plot workday hours
    for i, county in enumerate(county_names):
        years_data = list(yearly_data[county]['workday_hours'].keys())
        hours_data = list(yearly_data[county]['workday_hours'].values())
        
        line = ax2.plot(years_data, hours_data,
                marker='o', linewidth=2.5, markersize=8,
                label=f"{county} County", color=colors[i])[0]
        
        # Add data labels
        for x, y in zip(years_data, hours_data):
            ax2.annotate(f'{y}', 
                        (x, y),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9)
    
    ax2.set_title('Workday Hours (8am-6pm) of Power Outages per Year', fontsize=16)
    ax2.set_xlabel('Year', fontsize=14)
    ax2.set_ylabel('Workday Hours with Outages', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Plot average hours per home
    for i, county in enumerate(county_names):
        years_data = list(yearly_data[county]['avg_hours_per_home'].keys())
        hours_data = list(yearly_data[county]['avg_hours_per_home'].values())
        
        line = ax3.plot(years_data, hours_data,
                marker='o', linewidth=2.5, markersize=8,
                label=f"{county} County", color=colors[i])[0]
        
        # Add data labels
        for x, y in zip(years_data, hours_data):
            ax3.annotate(f'{y:.2f}', 
                        (x, y),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9)
    
    ax3.set_title('Average Hours of Power Outages per Home per Year', fontsize=16)
    ax3.set_xlabel('Year', fontsize=14)
    ax3.set_ylabel('Average Hours per Home', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)
    
    # Plot average workday hours per home
    for i, county in enumerate(county_names):
        years_data = list(yearly_data[county]['avg_workday_hours_per_home'].keys())
        hours_data = list(yearly_data[county]['avg_workday_hours_per_home'].values())
        
        line = ax4.plot(years_data, hours_data,
                marker='o', linewidth=2.5, markersize=8,
                label=f"{county} County", color=colors[i])[0]
        
        # Add data labels
        for x, y in zip(years_data, hours_data):
            ax4.annotate(f'{y:.2f}', 
                        (x, y),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9)
    
    ax4.set_title('Average Workday Hours (8am-6pm) of Power Outages per Home per Year', fontsize=16)
    ax4.set_xlabel('Year', fontsize=14)
    ax4.set_ylabel('Average Workday Hours per Home', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"outage_hours_analysis_{'_'.join(county_names)}.png", dpi=300, bbox_inches='tight')
    
    print(f"\nOutage hours analysis for {', '.join(county_names)} counties completed.")
    print(f"Results saved as 'outage_hours_analysis_{'_'.join(county_names)}.png'")
    return fig

def main():
    """Main function to run the script"""
    county_names = ["Fulton", "Washtenaw", "San Diego"]
    
    '''
    # 1. Full year analysis
    asPercentage = input("Graph as percent of customers affected or total customer count? (a/b): ")
    if asPercentage.lower() == 'a':
        asPercentage = True
    else:
        asPercentage = False
    print("Generating full year analysis...")
    plot_hourly_outages(county_names, asPercentage=asPercentage)
    
    # 2. Specific date range (example: summer months)
    print("Generating summer months analysis...")
    start_date = datetime(2023, 6, 1).date()
    end_date = datetime(2023, 8, 31).date()
    plot_hourly_outages(county_names, start_date, end_date, asPercentage)

    # 3. Historical trend analysis (2014-2023)
    print("Generating historical trend analysis (2014-2023)...")
    plot_historical_trends(county_names)
    '''

    # 4. Outage hours analysis
    print("Generating outage hours analysis...")
    analyze_outage_hours(county_names)

    #create_ev_outage_comparison()
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()