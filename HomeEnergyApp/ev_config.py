"""
EV configurations and utility functions for the Home Energy Optimizer.
Contains standard EV models, battery specifications, and charging profiles.
"""

# Dictionary of common EV models with their battery capacities and charging specs
EV_MODELS = {
    "Tesla Model 3 Standard Range": {
        "battery_capacity": 60.0,  # kWh
        "max_charge_rate_ac": 11.5,  # kW (Level 2)
        "max_discharge_rate": 10.0,  # kW (Vehicle-to-Grid/Home)
        "efficiency": 0.92,  # Round-trip efficiency
        "min_soc": 0.2,  # Minimum allowable state of charge
        "max_soc": 0.9,  # Maximum recommended state of charge
    },
    "Tesla Model Y Long Range": {
        "battery_capacity": 75.0,
        "max_charge_rate_ac": 11.5,
        "max_discharge_rate": 10.0,
        "efficiency": 0.92,
        "min_soc": 0.2,
        "max_soc": 0.9,
    },
    "Chevrolet Bolt": {
        "battery_capacity": 65.0,
        "max_charge_rate_ac": 7.2,
        "max_discharge_rate": 7.2,
        "efficiency": 0.9,
        "min_soc": 0.2,
        "max_soc": 0.9,
    },
    "Nissan Leaf": {
        "battery_capacity": 40.0,
        "max_charge_rate_ac": 6.6,
        "max_discharge_rate": 6.0,
        "efficiency": 0.89,
        "min_soc": 0.2,
        "max_soc": 0.9,
    },
    "Ford F-150 Lightning": {
        "battery_capacity": 98.0,
        "max_charge_rate_ac": 19.2,
        "max_discharge_rate": 9.6,
        "efficiency": 0.88,
        "min_soc": 0.2,
        "max_soc": 0.9,
    },
    "Hyundai Ioniq 5": {
        "battery_capacity": 77.4,
        "max_charge_rate_ac": 10.9,
        "max_discharge_rate": 10.0,
        "efficiency": 0.91,
        "min_soc": 0.2,
        "max_soc": 0.9,
    },
    "Rivian R1T": {
        "battery_capacity": 135.0,
        "max_charge_rate_ac": 17.6,
        "max_discharge_rate": 11.5,
        "efficiency": 0.89,
        "min_soc": 0.2,
        "max_soc": 0.9,
    },
    "Custom": {
        "battery_capacity": 60.0,
        "max_charge_rate_ac": 11.0,
        "max_discharge_rate": 11.0,
        "efficiency": 0.9,
        "min_soc": 0.2,
        "max_soc": 0.9,
    }
}

# Common utility rate structures by region
UTILITY_RATES = {
    "Fulton County (GA Power)": {
        "name": "Georgia Power Time-of-Use",
        "type": "time-of-use",
        "summer": {  # June-September
            "peak": 0.24,       # 2pm-7pm (weekdays)
            "off_peak": 0.07,   # All other times
        },
        "winter": {  # October-May
            "peak": 0.12,       # 7am-12pm (weekdays)
            "off_peak": 0.06,   # All other times
        }
    },
    "Washtenaw County (DTE)": {
        "name": "DTE Energy Time-of-Day",
        "type": "time-of-use",
        "summer": {  # June-September
            "peak": 0.18,       # 11am-7pm (weekdays)
            "mid_peak": 0.12,   # 7am-11am, 7pm-11pm (weekdays)
            "off_peak": 0.08,   # 11pm-7am (all days) and weekends
        },
        "winter": {  # October-May
            "peak": 0.15,       # 11am-7pm (weekdays)
            "mid_peak": 0.11,   # 7am-11am, 7pm-11pm (weekdays)
            "off_peak": 0.07,   # 11pm-7am (all days) and weekends
        }
    },
    "San Diego County (SDG&E)": {
        "name": "SDG&E Time-of-Use",
        "type": "time-of-use",
        "summer": {  # June-October
            "super_peak": 0.48,  # 4pm-9pm (all days)
            "peak": 0.30,        # 6am-4pm, 9pm-12am (all days)
            "off_peak": 0.25,    # 12am-6am (all days)
        },
        "winter": {  # November-May
            "super_peak": 0.35,  # 4pm-9pm (all days)
            "peak": 0.25,        # 6am-4pm, 9pm-12am (all days)
            "off_peak": 0.22,    # 12am-6am (all days)
        }
    },
    "Flat Rate": {
        "name": "Standard Flat Rate",
        "type": "flat",
        "rate": 0.13  # Flat rate all hours
    }
}

# Standard EV availability profiles
EV_AVAILABILITY_PROFILES = {
    "Home at Night": {
        "description": "EV is at home during evening/night hours (typical commuter)",
        "hours_available": list(range(0, 7)) + list(range(18, 24))  # 6pm-7am
    },
    "Work From Home": {
        "description": "EV is at home most of the day (work from home)",
        "hours_available": list(range(0, 24))  # All day
    },
    "Daytime Only": {
        "description": "EV is at home only during day hours (night shift worker)",
        "hours_available": list(range(8, 17))  # 8am-5pm
    },
    "Weekend Only": {
        "description": "EV is only available on weekends",
        "hours_available": []  # Will be handled separately in the UI
    },
    "Custom": {
        "description": "Custom availability profile",
        "hours_available": []  # Empty by default, to be set by user
    }
}

def get_utility_rate_schedule(utility_name, date=None, is_weekend=False):
    """
    Get hourly electricity rates for a specific utility and date.
    
    Parameters:
    utility_name (str): Name of the utility
    date (datetime, optional): Date to determine season (summer/winter)
    is_weekend (bool, optional): Whether the date is a weekend
    
    Returns:
    dict: Hourly electricity rates ($/kWh)
    """
    if utility_name not in UTILITY_RATES:
        # Return a default flat rate if utility not found
        return {hour: 0.13 for hour in range(24)}
    
    utility = UTILITY_RATES[utility_name]
    hourly_rates = {}
    
    # Handle flat rate
    if utility["type"] == "flat":
        return {hour: utility["rate"] for hour in range(24)}
    
    # For time-of-use, determine season
    # Default to summer if no date provided
    season = "summer"
    if date:
        month = date.month
        if utility_name == "Fulton County (GA Power)":
            season = "summer" if 6 <= month <= 9 else "winter"
        elif utility_name == "Washtenaw County (DTE)":
            season = "summer" if 6 <= month <= 9 else "winter"
        elif utility_name == "San Diego County (SDG&E)":
            season = "summer" if 6 <= month <= 10 else "winter"
    
    # Get the rates for the determined season
    rates = utility[season]
    
    # Assign hourly rates based on utility-specific time-of-use periods
    for hour in range(24):
        if utility_name == "Fulton County (GA Power)":
            if is_weekend:
                hourly_rates[hour] = rates["off_peak"]
            else:
                if season == "summer" and 14 <= hour < 19:  # 2pm-7pm peak in summer
                    hourly_rates[hour] = rates["peak"]
                elif season == "winter" and 7 <= hour < 12:  # 7am-12pm peak in winter
                    hourly_rates[hour] = rates["peak"]
                else:
                    hourly_rates[hour] = rates["off_peak"]
        
        elif utility_name == "Washtenaw County (DTE)":
            if is_weekend:
                hourly_rates[hour] = rates["off_peak"]
            else:
                if 11 <= hour < 19:  # 11am-7pm peak
                    hourly_rates[hour] = rates["peak"]
                elif 7 <= hour < 11 or 19 <= hour < 23:  # 7am-11am, 7pm-11pm mid-peak
                    hourly_rates[hour] = rates["mid_peak"]
                else:  # 11pm-7am off-peak
                    hourly_rates[hour] = rates["off_peak"]
        
        elif utility_name == "San Diego County (SDG&E)":
            if 16 <= hour < 21:  # 4pm-9pm super peak
                hourly_rates[hour] = rates["super_peak"]
            elif 6 <= hour < 16 or 21 <= hour < 24:  # 6am-4pm, 9pm-12am peak
                hourly_rates[hour] = rates["peak"]
            else:  # 12am-6am off-peak
                hourly_rates[hour] = rates["off_peak"]
    
    return hourly_rates 