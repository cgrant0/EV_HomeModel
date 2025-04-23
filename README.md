# Home Energy Model with EV Battery Optimization

This application models home energy usage combined with EV battery storage to optimize energy costs, prepare for grid outages, and reduce strain on the electrical grid. It allows users to analyze real energy and weather data, customize EV battery parameters, and simulate various optimization scenarios.

Additional python files provide context and deeper data analysis, the output of which can be seen in 'Output_Grahps'.

## Features

- **Data Integration**: Import and analyze real energy usage data, weather data, and power outage statistics from multiple cities
- **EV Battery Optimization**: Model how an EV battery can be used as home energy storage
- **Cost Savings Analysis**: Calculate potential savings from optimizing when to charge/discharge the EV battery
- **Outage Protection**: Model how EV batteries can provide backup power during grid outages
- **Customizable Parameters**: Adjust EV models, battery specs, utility rates, and availability schedules
- **Interactive UI**: View results through interactive plots and customizable options

## Data Sources

The application can work with multiple data sources:

1. **Energy Consumption Data**: 15-minute interval energy usage data for Atlanta, Ann Arbor, and San Diego (2018).
2. **Weather Data**: Historical temperature data aligned with the energy consumption data.
3. **Power Outage Data**: Historical outage statistics from 2014-2023
4. **EV Charging Patterns**: Standard EV charging time distributions
5. **Utility Rate Structures**: Actual utility rate plans based on location and time-of-use (only Fulton, San Diego, and Washtenaw counties)
6. **Data Expansion**: Energy consumption, weather, and power outage data is available for all counties in the US, allowing future expansion of tool.

## Requirements

See requirements.txt for exact versions.

- Python 3.8+
- PyQt5
- Matplotlib
- Pandas
- NumPy
- SciPy

## Installation

1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```-
3. Install all data files
- Weather and energy data should be within the repo
- Power outage data must be downloaded from https://figshare.com/s/417a4f147cf1357a5391?file=42547894
   - All outage files have been renamed, to 'outages_20XX.csv' so remove the 'eagleai_' from each file name
4. Ensure data files are correctly placed in their respective directories

## Usage

1. Run the application:
   ```
   python home_energy_app.py
   ```

2. In the "Data Selection" tab:
   - Select a city and load its data
   - Choose a specific date to analyze
   - Optionally load power outage data
   - View energy usage patterns for the selected day

3. In the "EV Configuration" tab:
   - Select an EV model or customize battery parameters
   - Set when the EV will be available at home
   - Configure battery behavior settings

4. In the "Optimization" tab:
   - Select utility rate structure
   - Set optimization priorities
   - Configure outage protection capacity
   - Run the optimization

5. In the "Results" tab:
   - View detailed breakdown of energy usage, battery state, and costs
   - See potential savings and battery cycling data
   - Export results for further analysis

## Project Structure

- `home_energy_app.py`: Main application with UI
- `data_manager.py`: Handles loading and processing various data sources
- `battery_optimizer.py`: Contains the optimization algorithm for battery usage
- `ev_config.py`: Stores EV configurations and utility rate structures
- Data directories:
  - `Atlanta/`, `Ann_Arbor/`, `San_Diego/`: City-specific energy and weather data
  - `Power_Data/`: Power outage statistics

## Customization

- **Custom EV Models**: Add your own EV with specific battery parameters
- **Utility Rates**: Configure different time-of-use pricing structures
- **Energy Profiles**: Create custom energy usage patterns
- **Optimization Priorities**: Balance between cost savings and outage protection (in beta)

## Current Issues
- Date resets after loading city data on first page
- Outage data loads for all counties, takes a few seconds (not necessary)
- Rate graph doesn't display for ATL on page 3 unless another selection is made first then returned to ATL
- Legends won't populate for graphs 1 and 3 on the results page
- Minor error with plotting bars on graphs 1 and 3 on results page
- Prioritize cost savings silder and outage reserve implementation not finished (not connected to optimizer.py)
- Still a rudimentary tool, needs more flushing out

## Extending the Model

The application is designed with modularity and expandability in mind:

- Add more cities and data sources by extending the `DataManager` class
- Implement new optimization strategies in `BatteryOptimizer`
- Add more EV models and utility rate structures in `ev_config.py`
- Extend the UI with additional features in `home_energy_app.py`

## Future Direction

Given the ability to contiune working on the tool, there are a few areas to improve:

- Optimization algorithm: Currently, it's just a greedy approach based on load shedding, with no abality to augment optimization style. Creating multiple approaches and including ancillary costs, stipends from utility companies, and power outage savings will provide better estimates.
- Vehicle charging data: Charging data is constant for all locations and most likely not emulative of real-life charging patterns. Anecdotal evidence points to charging once every few days rather than every day.
- Data management: All data must be stored locally for the app to be run, if multiple people are to use the app moving forward, the energy and power outage data must be ported elsewhere.
- Data display: There's a bit too much going on at the moment, prioritize only relevant battery data moving forward. 
- Long term ideas: Matter system integration, better power company and pricing model information, exact battery characteristics from Hyundai or elsewhere, more exploration of custom energy use profiles. 

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Energy data sourced from public utilities
- Weather data from NOAA
- Power outage data from the Homeland Infrastructure Foundation-Level Data (HIFLD)