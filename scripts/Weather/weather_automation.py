import requests
import json
from datetime import datetime
import pytz

"""
This script is used to generate a weather JSON file for the Unreal Engine.
It uses the OpenWeatherMap API to get the current weather data and the Open-Meteo API to get the historical weather data.
The weather data is then mapped to a JSON structure and saved to a file.
"""

# Function to get weather data from OpenWeatherMap API
def get_weather_data_openweathermap(api_key, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,alerts&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching weather data from OpenWeatherMap: {response.status_code}")

# Function to get historical weather data from Open-Meteo API
def get_weather_data_open_meteo(lat, lon, date):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}&hourly=temperature_2m,cloudcover,windspeed_10m,winddirection_10m"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching weather data from Open-Meteo: {response.status_code}")

# Function to map OpenWeatherMap data to JSON structure
def map_weather_to_json_openweathermap(weather_data, time_of_day_minutes):
    current_data = weather_data.get('current', {})
    clouds = current_data.get('clouds', 0) / 100.0  # Normalizing cloud coverage to a [0, 1] range
    wind_speed = current_data.get('wind_speed', 0)
    wind_direction = current_data.get('wind_deg', 0)
    weather_condition = current_data.get('weather', [{}])[0].get('main', 'Clear')
    
    # Mapping the weather conditions to weather presets
    weather_presets = {
        'Clear': 'Clear_Skies',
        'Clouds': 'Cloudy',
        'Rain': 'Rainy',
        'Snow': 'Snowy',
        'Fog': 'Foggy'
    }
    weather_preset = weather_presets.get(weather_condition, 'Clear_Skies')

    stars_intensity = 1.0 if time_of_day_minutes > 1200 or time_of_day_minutes < 300 else 0.2

    json_data = {
        "TimeOfDay": time_of_day_minutes,
        "CloudCoverage": clouds * 10,  # Adjusting to a scale
        "CloudSpeed": wind_speed / 10,  # Assuming some logic to cloud speed
        "Fog": clouds * 5,  # Assuming fog increases with cloud coverage
        "CloudShadowEnable": True if clouds > 0.3 else False,
        "WeatherPreset": weather_preset,
        "WindDirection": wind_direction,
        "NightBrightness": 0.5 if stars_intensity == 1.0 else 1.0,
        "Dust": 0.1,
        "stars_intensity": stars_intensity
    }

    return json_data

# Function to map Open-Meteo data to your JSON structure
def map_weather_to_json_open_meteo(weather_data, time_of_day_minutes):
    hourly_data = weather_data.get('hourly', {})
    clouds = hourly_data.get('cloudcover', [0])[0] / 100.0  # Normalize cloud coverage to [0, 1] range
    wind_speed = hourly_data.get('windspeed_10m', [0])[0]
    wind_direction = hourly_data.get('winddirection_10m', [0])[0]

    stars_intensity = 1.0 if time_of_day_minutes > 1200 or time_of_day_minutes < 300 else 0.2

    json_data = {
        "TimeOfDay": time_of_day_minutes,
        "CloudCoverage": clouds * 10,
        "CloudSpeed": wind_speed / 10,
        "Fog": clouds * 5,
        "CloudShadowEnable": True if clouds > 0.3 else False,
        "WeatherPreset": 'Clear_Skies',  # Open-Meteo does not provide precise weather conditions
        "WindDirection": wind_direction,
        "NightBrightness": 0.5 if stars_intensity == 1.0 else 1.0,
        "Dust": 0.1,
        "stars_intensity": stars_intensity
    }

    return json_data

# Function to convert the current time of day to minutes
def get_time_of_day_minutes(time, date, timezone):
    local_time = timezone.localize(datetime.combine(date, time))
    total_minutes = local_time.hour * 60 + local_time.minute
    return total_minutes

# Main function to generate JSON
def generate_weather_json(api_key, lat, lon, time, date, timezone_str, api_choice='openweathermap'):
    # Convert timezone string to timezone object
    timezone = pytz.timezone(timezone_str)

    time_of_day_minutes = get_time_of_day_minutes(time, date, timezone)

    if api_choice == 'openweathermap':
        # Fetch the weather data from OpenWeatherMap
        weather_data = get_weather_data_openweathermap(api_key, lat, lon)
        # Map the weather data to JSON structure
        weather_json = map_weather_to_json_openweathermap(weather_data, time_of_day_minutes)
    elif api_choice == 'openmeteo':
        # Fetch the historical weather data from Open-Meteo
        weather_data = get_weather_data_open_meteo(lat, lon, date)
        # Map the weather data to  JSON structure
        weather_json = map_weather_to_json_open_meteo(weather_data, time_of_day_minutes)
    else:
        raise ValueError("Invalid API choice. Choose either 'openweathermap' or 'openmeteo'.")

    # Output the JSON file
    with open('weather_config.json', 'w') as json_file:
        json.dump(weather_json, json_file, indent=4)

    print("Generated JSON:", json.dumps(weather_json, indent=4))

api_key = 'API_KEY'  
lat = 33.44  #
lon = -94.04  
time = datetime.now().time()  
date = datetime.now().date()  
timezone_str = 'America/Chicago'  
api_choice = 'openmeteo'  # Choose between 'openweathermap' and 'openmeteo'

# Generate the JSON file
generate_weather_json(api_key, lat, lon, time, date, timezone_str, api_choice)
