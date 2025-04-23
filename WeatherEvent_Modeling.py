import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# City data dictionary for easy reference
CITY_DATA = {
    'Atlanta': {
        'dwellings': 449880,
        'weather_file': 'Atlanta/atlantaairportweather_72219013874.csv',
        'energy_file': 'Atlanta/Atlanta_15_minute_timeseries_data.csv'
    },
    'Ann Arbor': {
        'dwellings': 149394,
        'weather_file': 'Ann Arbor/detroitairportweather_72537094847.csv',
        'energy_file': 'Ann Arbor/AnnArbor_15_minute_timeseries_data_year.csv'
    },
    'San Diego': {
        'dwellings': 1187651,
        'weather_file': 'San Diego/sandiegoairportweather_72290023188.csv',
        'energy_file': 'San Diego/SanDiego_15_minute_timeseries_data.csv'
    }
}

def load_city_data(city_name):
    """
    Load and prepare weather and energy data for a specific city
    
    Parameters:
    city_name (str): Name of the city ('Atlanta', 'Ann Arbor', or 'San Diego')
    
    Returns:
    tuple: (weather_data, energy_data, number_of_dwellings)
    """
    if city_name not in CITY_DATA:
        raise ValueError(f"City {city_name} not supported. Choose from: {list(CITY_DATA.keys())}")
    
    city_info = CITY_DATA[city_name]
    
    # Load data files
    weather_data = pd.read_csv(city_info['weather_file'], low_memory=False)
    energy_data = pd.read_csv(city_info['energy_file'])
    
    # Clean and process the data
    weather_clean = clean_weather_data(weather_data)
    energy_clean = process_energy_data(energy_data, city_info['dwellings'])
    
    return weather_clean, energy_clean, city_info['dwellings']

def clean_weather_data(weather_df):
    """
    Clean and process weather data
    
    Parameters:
    weather_df (DataFrame): Raw weather data
    
    Returns:
    DataFrame: Cleaned weather data with datetime and temperature columns
    """
    # Convert temperature from SYNOP format to Celsius
    # In SYNOP, temperature is in tenths of degrees C with leading sign
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
    Process energy data and calculate load per dwelling
    
    Parameters:
    energy_df (DataFrame): Raw energy data
    num_dwellings (int): Number of dwellings in the city
    
    Returns:
    DataFrame: Processed energy data with total load and load per dwelling
    """
    # Convert timestamp to datetime
    energy_df['datetime'] = pd.to_datetime(energy_df['Timestamp (EST)']).dt.tz_localize(None)
    
    energy_df['total_load'] = energy_df['baseline.out.electricity.total.energy_consumption.kwh']
    
    # Normalize per dwelling
    energy_df['load_per_dwelling'] = energy_df['total_load'] / num_dwellings
    
    return energy_df[['datetime', 'total_load', 'load_per_dwelling']]

def analyze_temperature_load_relationship(weather_df, energy_df):
    """
    Analyze the relationship between temperature and energy load
    
    Parameters:
    weather_df (DataFrame): Cleaned weather data
    energy_df (DataFrame): Processed energy data
    
    Returns:
    tuple: (combined_data, coefficients, r_squared, model, poly_features)
    """
    # Resample energy data to hourly means
    energy_hourly = energy_df.set_index('datetime').resample('h').mean()

    # Resample weather data to match energy timestamps using nearest neighbor interpolation
    weather_hourly = (
        weather_df.set_index('datetime')
        .resample('h')  # Resample to hourly frequency
        .mean()  # Average temperatures within each hour
    )

    # Merge datasets on the hourly-aligned index
    combined_df = pd.merge(
        energy_hourly, 
        weather_hourly, 
        left_index=True, 
        right_index=True, 
        how='inner'
    )

    X = combined_df['temperature_c'].values.reshape(-1, 1)
    y = combined_df['load_per_dwelling'].values
    
    # Create polynomial features (degree=2 for quadratic)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    
    # Fit quadratic regression
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Calculate R-squared
    y_pred = model.predict(X_poly)
    r_squared = r2_score(y, y_pred)
    
    # Get coefficients (a, b, c) for ax² + bx + c
    a, b, c = model.coef_[2], model.coef_[1], model.intercept_
    
    return combined_df, (a, b, c), r_squared, model, poly_features

def plot_and_analyze_city(city_name):
    """
    Load, analyze, and visualize data for a specific city
    
    Parameters:
    city_name (str): Name of the city to analyze
    
    Returns:
    tuple: (coefficients, r_squared, temperature_of_minimum_usage)
    """
    # Load and process city data
    weather_data, energy_data, num_dwellings = load_city_data(city_name)
    
    # Analyze relationship
    combined_data, coefficients, r_squared, model, poly_features = analyze_temperature_load_relationship(
        weather_data, energy_data
    )
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.scatter(combined_data['temperature_c'], combined_data['load_per_dwelling'], 
               alpha=0.5, s=1, label='Actual Data')

    # Generate smooth curve for quadratic fit
    X_smooth = np.linspace(combined_data['temperature_c'].min(), 
                          combined_data['temperature_c'].max(), 
                          300).reshape(-1, 1)
    y_smooth = model.predict(poly_features.transform(X_smooth))

    plt.plot(X_smooth, y_smooth, 'r', linewidth=2, 
             label=f'Quadratic regression\nR² = {r_squared:.3f}')

    plt.xlabel('Temperature (°C)')
    plt.ylabel('Energy Consumption per Dwelling (kWh)')
    plt.title(f'Temperature vs. Energy Consumption Relationship in {city_name} (2018)')
    plt.legend()
    plt.show()

    # Print results
    a, b, c = coefficients
    print(f"\nModel Results for {city_name}:")
    print(f"Quadratic equation: y = {a:.6f}x² + {b:.6f}x + {c:.6f}")
    print(f"Where y is energy consumption in kWh per dwelling and x is temperature in °C")
    print(f"R-squared value: {r_squared:.3f}")

    # Calculate temperature of minimum energy usage (vertex of parabola)
    temp_min = -b / (2*a)
    print(f"Minimum energy usage occurs at {temp_min:.1f}°C")
    
    return coefficients, r_squared, temp_min

def compare_all_cities():
    """
    Analyze all cities and create a comparison plot
    """
    plt.figure(figsize=(14, 10))
    
    # Colors for each city
    colors = {
        'Atlanta': 'red',
        'Ann Arbor': 'blue',
        'San Diego': 'green'
    }
    
    results = {}
    
    for city in CITY_DATA.keys():
        # Load and process city data
        weather_data, energy_data, num_dwellings = load_city_data(city)
        
        # Analyze relationship
        combined_data, coefficients, r_squared, model, poly_features = analyze_temperature_load_relationship(
            weather_data, energy_data
        )
        
        # Store results
        a, b, c = coefficients
        temp_min = -b / (2*a)
        results[city] = {
            'coefficients': coefficients,
            'r_squared': r_squared,
            'temp_min': temp_min,
            'data': combined_data,
            'model': model,
            'poly_features': poly_features
        }
        
        # Plot data points with small alpha for visualization
        #plt.scatter(combined_data['temperature_c'], combined_data['load_per_dwelling'], 
                   #alpha=0.2, s=1, color=colors[city], label=f'{city} Data')
        
        # Generate smooth curve for quadratic fit
        X_smooth = np.linspace(combined_data['temperature_c'].min(), 
                              combined_data['temperature_c'].max(), 
                              300).reshape(-1, 1)
        y_smooth = model.predict(poly_features.transform(X_smooth))

        plt.plot(X_smooth, y_smooth, color=colors[city], linewidth=2, 
                 label=f'{city} Model (R² = {r_squared:.3f})')
        
        # Mark minimum point
        min_y = a * (temp_min ** 2) + b * temp_min + c
        #plt.plot(temp_min, min_y, 'o', color=colors[city], markersize=8)
        #plt.annotate(f'{temp_min:.1f}°C', 
                     #xy=(temp_min, min_y),
                     #xytext=(temp_min + 1, min_y + 0.005),
                     #color=colors[city],
                     #fontweight='bold')

    plt.xlabel('Temperature (°C)')
    plt.ylabel('Energy Consumption per Dwelling (kWh)')
    plt.title('Temperature vs. Energy Consumption Relationship Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print comparison results
    print("\nComparison of Minimum Energy Usage Temperatures:")
    for city, data in results.items():
        print(f"{city}: {data['temp_min']:.1f}°C (R² = {data['r_squared']:.3f})")
    
    return results

# To analyze a single city:
#plot_and_analyze_city('Atlanta')
#plot_and_analyze_city('Ann Arbor')
#plot_and_analyze_city('San Diego')

# To compare all cities:
compare_all_cities()