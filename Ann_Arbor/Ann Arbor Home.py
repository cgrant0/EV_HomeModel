import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg

AnnA_15min_year_data = pd.read_csv('/Users/zackburkhardt/Documents/Spring 2025/Hyundai Research/Data/Best Data/Ann Arbor/AnnArbor_15_minute_timeseries_data_year.csv', low_memory=False)
AnnA_dwell = pd.read_csv('/Users/zackburkhardt/Documents/Spring 2025/Hyundai Research/Data/Best Data/Ann Arbor/AnnArbor_dwellingunits.csv')

AnnA_dwelling_units = AnnA_dwell.iloc[:, 3].sum()

# Load CSV without automatic parsing
# Convert the timestamp column explicitly
AnnA_15min_year_data["Timestamp (EST)"] = pd.to_datetime(
    AnnA_15min_year_data["Timestamp (EST)"], errors="coerce"
)
# Set as index
AnnA_15min_year_data.set_index("Timestamp (EST)", inplace=True)
specific_day = AnnA_15min_year_data.loc["2018-01-02"]

# Remove timezone information
AnnA_15min_year_data.index = AnnA_15min_year_data.index.tz_localize(None)
# Extract Year-Month
AnnA_15min_year_data["Year-Month"] = AnnA_15min_year_data.index.to_period("M")
electricity_columns = [col for col in AnnA_15min_year_data.columns if 'electricity' in col.lower()]

# Function to extract the end-use name
def clean_column_name(name):
    parts = name.split(".")  # Split by dots
    for part in parts:
        if part not in ["baseline", "out", "electricity", "energy", "consumption", "kwh"]:
            return part.replace("_", " ")  # Replace underscores with spaces
    return name  # Fallback in case no match is found

# Apply the function to rename all columns
AnnA_15min_year_data.rename(columns={col: clean_column_name(col) for col in AnnA_15min_year_data.columns}, inplace=True)

# Print to check
monthly_usage_total = AnnA_15min_year_data.groupby("Year-Month").sum()
monthly_usage = monthly_usage_total / AnnA_dwelling_units
unwanted_keywords = ["wood", "natural gas", "propane", "fuel oil", "site energy"]
filtered_columns = [col for col in monthly_usage.columns if not any(keyword in col.lower() for keyword in unwanted_keywords)]
monthly_usage_filtered = monthly_usage[filtered_columns]

top_10_end_uses = monthly_usage_filtered.sum().nlargest(6).index  # Find the 6 highest-consuming columns
monthly_usage_top10 = monthly_usage_filtered[top_10_end_uses]  # Select only those columns
# Convert PeriodIndex to string and then to month name
monthly_usage.index = monthly_usage.index.strftime('%B')
# Ensure the index is correctly formatted as strings before plotting
monthly_usage_top10.index = monthly_usage_top10.index.strftime('%B')
monthly_usage_top10 = monthly_usage_top10.iloc[:-1]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Get the default color cycle
color_order = [colors[2], colors[4], colors[3], colors[0], colors[1], colors[5]]  # Custom order
monthly_usage_top10.plot(kind="bar", figsize=(12, 6), width=0.8,color = color_order)
plt.xlabel("Month")
plt.ylabel("Electricity Usage (kWh)")
plt.title("Ann Arbor Monthly Electricity Usage per Home 2018")

# Move the legend outside the plot area
plt.legend(title="End-Use", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xticks(rotation=45)

# Adjust the layout to make room for the legend
plt.tight_layout()

# Show the plot
plt.show()


# Resample to hourly intervals (sum up 15-minute data)
hourly_data = specific_day.resample("H").sum()
specific_day = hourly_data
specific_day.rename(columns={col: clean_column_name(col) for col in specific_day.columns}, inplace=True)

# Print to check
daily_usage = specific_day / AnnA_dwelling_units
unwanted_keywords = ["wood", "natural gas", "propane", "fuel oil", "site energy", 'total']
filtered_columns = [col for col in daily_usage.columns if not any(keyword in col.lower() for keyword in unwanted_keywords)]
daily_usage_filtered = daily_usage[filtered_columns]

top_10_end_uses = daily_usage_filtered.sum().nlargest(5).index  # Find the 6 highest-consuming columns
daily_usage_top10 = daily_usage_filtered[top_10_end_uses]  # Select only those columns
# Convert PeriodIndex to string and then to month name
daily_usage.index = daily_usage.index.strftime('%H')
# Ensure the index is correctly formatted as strings before plotting
daily_usage_top10.index = daily_usage_top10.index.strftime('%H')
daily_usage_top10 = daily_usage_top10.iloc[:-1]





color_order_new = [colors[3], colors[4], colors[7], colors[1], colors[8]]  # Custom order
daily_usage_top10.plot(kind="bar", figsize=(10, 5), width=0.8,color = color_order_new)

plt.xlabel("Hour of the Day")
plt.ylabel("Electricity Usage (kWh)")
plt.title("Ann Arbor Hourly Electricity Usage on January 2nd, 2018")
plt.xticks(rotation=0)  # Keep hours readable
plt.grid(axis="y")
# Move the legend outside the plot area
plt.legend(title="End-Use", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
