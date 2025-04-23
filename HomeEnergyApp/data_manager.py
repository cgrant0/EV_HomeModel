import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataManager:
    """
    Manages data access for the Home Energy Optimizer.
    Centralizes loading and preprocessing of energy, weather, outage and EV data.
    """
    
    # City data dictionary for easy reference
    CITY_DATA = {
        'Atlanta': {
            'dwellings': 449880,
            'weather_file': 'Atlanta/atlantaairportweather_72219013874.csv',
            'energy_file': 'Atlanta/Atlanta_15_minute_timeseries_data.csv',
            'county': 'Fulton'
        },
        'Ann Arbor': {
            'dwellings': 149394,
            'weather_file': 'Ann_Arbor/detroitairportweather_72537094847.csv',
            'energy_file': 'Ann_Arbor/AnnArbor_15_minute_timeseries_data_year.csv',
            'county': 'Washtenaw'
        },
        'San Diego': {
            'dwellings': 1187651,
            'weather_file': 'San_Diego/sandiegoairportweather_72290023188.csv',
            'energy_file': 'San_Diego/SanDiego_15_minute_timeseries_data.csv',
            'county': 'San Diego'
        }
    }
    
    # Standard EV charging pattern data (percentage of EVs charging by hour)
    EV_CHARGING_DATA = {
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
    
    def __init__(self):
        """Initialize the data manager"""
        self.weather_data = {}
        self.energy_data = {}
        self.outage_data = {}
        self.ev_charging_data = pd.DataFrame({
            'Hour': list(self.EV_CHARGING_DATA.keys()), 
            'Charging_Percentage': list(self.EV_CHARGING_DATA.values())
        })
        
    def load_city_data(self, city_name):
        """
        Load weather and energy data for a specific city
        
        Parameters:
        city_name (str): Name of the city ('Atlanta', 'Ann Arbor', or 'San Diego')
        
        Returns:
        bool: Success status
        """
        if city_name not in self.CITY_DATA:
            print(f"City {city_name} not supported. Choose from: {list(self.CITY_DATA.keys())}")
            return False
        
        city_info = self.CITY_DATA[city_name]
        
        try:
            # Load weather data
            weather_data = pd.read_csv(city_info['weather_file'], low_memory=False)
            self.weather_data[city_name] = self._clean_weather_data(weather_data)
            
            # Load energy data
            energy_data = pd.read_csv(city_info['energy_file'])
            self.energy_data[city_name] = self._process_energy_data(energy_data, city_info['dwellings'])
            
            return True
        except Exception as e:
            print(f"Error loading data for {city_name}: {str(e)}")
            return False
    
    def _clean_weather_data(self, weather_df):
        """Clean and process weather data"""
        # Convert temperature from SYNOP format to Celsius
        weather_df['temperature_c'] = weather_df['TMP'].apply(
            lambda x: (float(str(x)[1:-2]) / 10 * (-1 if str(x)[0] == '-' else 1)) if pd.notnull(x) else np.nan
        )
        
        # Remove rows where temperature_c is 999.9
        weather_df = weather_df.loc[weather_df['temperature_c'] != 999.9]
    
        # Convert to datetime
        weather_df['datetime'] = pd.to_datetime(weather_df['DATE']).dt.tz_localize(None)
        return weather_df[['datetime', 'temperature_c']].dropna()
    
    def _process_energy_data(self, energy_df, num_dwellings):
        """Process energy data and calculate load per dwelling"""
        # Convert timestamp to datetime
        energy_df['datetime'] = pd.to_datetime(energy_df['Timestamp (EST)']).dt.tz_localize(None)
        
        energy_df['total_load'] = energy_df['baseline.out.electricity.total.energy_consumption.kwh']
        
        # Normalize per dwelling
        energy_df['load_per_dwelling'] = energy_df['total_load'] / num_dwellings
        
        return energy_df[['datetime', 'total_load', 'load_per_dwelling']]
    
    def load_outage_data(self, year=2023):
        """Load power outage data for a specific year"""
        try:
            outages_df = pd.read_csv(f'Power_Data/outages_{year}.csv')
            customers_df = pd.read_csv('Power_Data/county_FIPS_customers.csv')
            
            # Convert run_start_time to datetime
            outages_df['run_start_time'] = pd.to_datetime(outages_df['run_start_time'])
            
            # Extract hour from timestamp
            outages_df['hour'] = outages_df['run_start_time'].dt.hour
            
            # Add date column for filtering
            outages_df['date'] = outages_df['run_start_time'].dt.date
            
            self.outage_data = {
                'outages': outages_df,
                'customers': customers_df,
                'year': year
            }
            
            return True
        except Exception as e:
            print(f"Error loading outage data for year {year}: {str(e)}")
            return False
    
    def get_county_outage_data(self, county_name, date=None):
        """Get outage data for a specific county and optional date"""
        if not self.outage_data:
            print("Outage data not loaded. Call load_outage_data() first.")
            return None
        
        # Filter outages data for the specified county
        county_outages = self.outage_data['outages'][self.outage_data['outages']['county'] == county_name]
        
        if county_outages.empty:
            print(f"No outage data found for {county_name} County")
            return None
        
        # Get the FIPS code for this county
        fips_code = str(county_outages['fips_code'].iloc[0])
        
        # Get total customer count for this county
        customers_df = self.outage_data['customers']
        county_customers = customers_df[customers_df['County_FIPS'] == fips_code]
        
        if county_customers.empty:
            print(f"No customer data found for {county_name} County (FIPS: {fips_code})")
            return None
        
        total_customers = county_customers['Customers'].iloc[0]
        
        # Filter by date if provided
        if date:
            date_mask = county_outages['date'] == date
            filtered_outages = county_outages[date_mask]
            
            if filtered_outages.empty:
                print(f"No outage data for {county_name} County on {date}")
                return None
        else:
            filtered_outages = county_outages
        
        # Determine the correct column name based on year
        outage_column = 'sum' if self.outage_data['year'] == 2023 else 'customers_out'
        
        # Group by hour and calculate average outages
        hourly_avg = filtered_outages.groupby('hour')[outage_column].mean()
        
        # Calculate percentage of customers affected
        hourly_percentages = (hourly_avg / total_customers) * 100
        
        # Convert to DataFrame with all hours
        hours_df = pd.DataFrame({'hour': range(24)})
        outage_df = pd.DataFrame({
            'hour': hourly_percentages.index,
            'outage_percentage': hourly_percentages.values
        })
        
        result = pd.merge(hours_df, outage_df, on='hour', how='left').fillna(0)
        result.rename(columns={'hour': 'Hour'}, inplace=True)
        
        return result
    
    def get_energy_for_date(self, city_name, date):
        """Get hourly energy data for a specific city and date"""
        if city_name not in self.energy_data:
            print(f"Energy data for {city_name} not loaded. Call load_city_data() first.")
            return None
        
        energy_df = self.energy_data[city_name]
        
        # Filter by date
        date_mask = energy_df['datetime'].dt.date == date
        filtered_energy = energy_df[date_mask]
        
        if filtered_energy.empty:
            print(f"No energy data for {city_name} on {date}")
            return None
        
        # Resample to hourly data
        hourly_energy = filtered_energy.set_index('datetime').resample('H').mean()
        hourly_energy['hour'] = hourly_energy.index.hour
        
        return hourly_energy.reset_index()[['hour', 'load_per_dwelling']].rename(columns={'hour': 'Hour'})
    
    def get_weather_for_date(self, city_name, date):
        """Get hourly weather data for a specific city and date"""
        if city_name not in self.weather_data:
            print(f"Weather data for {city_name} not loaded. Call load_city_data() first.")
            return None
        
        weather_df = self.weather_data[city_name]
        
        # Filter by date
        date_mask = weather_df['datetime'].dt.date == date
        filtered_weather = weather_df[date_mask]
        
        if filtered_weather.empty:
            print(f"No weather data for {city_name} on {date}")
            return None
        
        # Resample to hourly data if there are multiple readings per hour
        hourly_weather = filtered_weather.set_index('datetime').resample('H').mean()
        hourly_weather['hour'] = hourly_weather.index.hour
        
        return hourly_weather.reset_index()[['hour', 'temperature_c']].rename(columns={'hour': 'Hour'})
    
    def get_ev_charging_pattern(self):
        """Get standard EV charging pattern data"""
        return self.ev_charging_data
    
    def get_available_dates(self, city_name):
        """Get a list of dates with available energy data for a city"""
        if city_name not in self.energy_data:
            print(f"Energy data for {city_name} not loaded. Call load_city_data() first.")
            return []
        
        energy_df = self.energy_data[city_name]
        dates = sorted(energy_df['datetime'].dt.date.unique())
        return dates
    
    def get_city_county(self, city_name):
        """Get the county name for a given city"""
        if city_name not in self.CITY_DATA:
            return None
        return self.CITY_DATA[city_name]['county'] 