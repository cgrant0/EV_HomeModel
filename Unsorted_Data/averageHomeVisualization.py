import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_prepare_data():
    # Read the TSV file
    df = pd.read_csv('ResStockData/metadata.tsv', sep='\t')
    
    # Define county codes and their readable names
    county_mapping = {
        'G1301210': 'Fulton, GA',
        'G2601610': 'Ann Arbor, MI',
        'G0600730': 'San Diego, CA'
    }
    
    # Filter for our counties of interest
    county_data = df[df['in.nhgis_county_gisjoin'].isin(county_mapping.keys())]
    
    # Filter for single-family detached homes
    sf_data = county_data[county_data['in.geometry_building_type_acs'] == 'Single-Family Detached']
    
    # Map county codes to readable names
    sf_data['county_name'] = sf_data['in.nhgis_county_gisjoin'].map(county_mapping)
    
    return sf_data

def create_home_size_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='county_name', y='in.floor_area_conditioned_ft_2', data=df)
    plt.title('Distribution of Single-Family Home Sizes by County')
    plt.ylabel('Square Feet')
    plt.xlabel('County')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Week1_Data/sf_home_sizes_county.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_home_age_distribution(df):
    plt.figure(figsize=(12, 6))
    age_dist = df.groupby(['county_name', 'in.vintage']).size().unstack()
    age_dist.plot(kind='bar', stacked=True)
    plt.title('Distribution of Single-Family Home Age by County')
    plt.xlabel('County')
    plt.ylabel('Count')
    plt.legend(title='Year Built', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Week1_Data/sf_home_ages_county.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_heating_system_distribution(df):
    plt.figure(figsize=(12, 6))
    heating_types = df.groupby(['county_name', 'in.hvac_heating_type']).size().unstack()
    heating_types.plot(kind='bar', stacked=True)
    plt.title('Heating System Types in Single-Family Homes by County')
    plt.xlabel('County')
    plt.ylabel('Count')
    plt.legend(title='Heating System', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Week1_Data/sf_heating_systems_county.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_cooling_system_distribution(df):
    plt.figure(figsize=(12, 6))
    cooling_types = df.groupby(['county_name', 'in.hvac_cooling_type']).size().unstack()
    cooling_types.plot(kind='bar', stacked=True)
    plt.title('Cooling System Types in Single-Family Homes by County')
    plt.xlabel('County')
    plt.ylabel('Count')
    plt.legend(title='Cooling System', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Week1_Data/sf_cooling_systems_county.png', bbox_inches='tight', dpi=300)
    plt.close()

def calculate_county_averages(df):
    # Calculate average metrics by county
    county_averages = df.groupby('county_name').agg({
        'in.floor_area_conditioned_ft_2': 'mean',
        'in.bedrooms': 'mean',
        'in.geometry_stories': 'mean',
        'in.window_area_ft_2': 'mean',
        'in.geometry_garage': lambda x: x.value_counts().index[0]  # Most common garage type
    }).round(2)
    
    # Add energy consumption averages
    county_averages['avg_electricity_consumption'] = df.groupby('county_name')['out.electricity.total.energy_consumption'].mean().round(2)
    county_averages['avg_natural_gas_consumption'] = df.groupby('county_name')['out.natural_gas.total.energy_consumption'].mean().round(2)
    
    # Add count of homes per county
    county_averages['number_of_homes'] = df.groupby('county_name').size()
    
    # Rename the garage column for clarity
    county_averages = county_averages.rename(columns={'in.geometry_garage': 'most_common_garage_type'})
    
    # Add garage type distribution as percentages
    for county in county_averages.index:
        county_data = df[df['county_name'] == county]
        garage_dist = county_data['in.geometry_garage'].value_counts(normalize=True) * 100
        for garage_type, percentage in garage_dist.items():
            county_averages.loc[county, f'garage_{garage_type}_pct'] = round(percentage, 1)
    
    # Calculate average home age and add it to the DataFrame
    current_year = datetime.now().year
    df['home_age'] = current_year - df['in.vintage']
    avg_home_age = df.groupby('county_name')['home_age'].mean().round(2)
    county_averages['avg_home_age'] = avg_home_age
    
    return county_averages

def main():
    # Load and prepare the data
    df = load_and_prepare_data()
    
    # Create individual visualizations
    create_home_size_distribution(df)
    create_home_age_distribution(df)
    create_heating_system_distribution(df)
    create_cooling_system_distribution(df)
    
    # Calculate and display county averages
    averages = calculate_county_averages(df)
    print("\nSingle-Family Home Averages by County:")
    print(averages)
    
    # Save averages to CSV
    averages.to_csv('Week1_Data/single_family_averages_by_county.csv')

if __name__ == "__main__":
    main()