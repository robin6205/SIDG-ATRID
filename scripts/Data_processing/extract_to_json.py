import csv
import json
import math

def lat_long_to_local_xy_rotated(target_lat, target_long, origin_lat, origin_long, scale=1):
    """
    Adjusted to account for 90 degrees rotation: Positive Y is East, Positive X is North.
    
    Parameters:
    - target_lat, target_long: Latitude and longitude of the target point.
    - origin_lat, origin_long: Latitude and longitude of the reference origin point.
    - scale: Scale factor for conversion to Unreal Engine units (default is 1, considering rotation).
    
    Returns:
    - Tuple (x, y) representing local coordinates in Unreal Engine environment.
    """
    R = 6378137  # Earth's radius in meters
    delta_lat = math.radians(target_lat - origin_lat)
    delta_long = math.radians(target_long - origin_long)
    avg_lat = math.radians((origin_lat + target_lat) / 2.0)
    
    x_distance = R * delta_lat  # X rotation
    y_distance = R * delta_long * math.cos(avg_lat)  # Longitude difference affects Y
    
    x = x_distance * scale
    y = y_distance * scale
    
    return x, y

# Function to clean NUL characters from a file
def remove_nul_characters(file_path):
    cleaned_lines = []
    with open(file_path, mode='rb') as file:
        for line in file:
            cleaned_line = line.replace(b'\x00', b'')  # Remove NUL characters
            cleaned_lines.append(cleaned_line)
    return cleaned_lines

# Read and process the CSV file
csv_file_path = 'output.csv'
waypoints = []

# Assuming you have the origin_latitude and origin_longitude from your Unreal Engine reference point
origin_lat = 40.284732  # Replace with your actual origin latitude
origin_long = -86.853282  # Replace with your actual origin longitude
scale = 100.5  # Adjust the scale if needed
offset = 22188  # Adjust the offset if needed

# Step 1: Clean NUL characters from the CSV file
cleaned_csv_data = remove_nul_characters(csv_file_path)

# Step 2: Write the cleaned data to a temporary file
temp_csv_file_path = 'cleaned_output.csv'
with open(temp_csv_file_path, mode='wb') as temp_file:
    temp_file.writelines(cleaned_csv_data)

# Step 3: Process the cleaned CSV file
with open(temp_csv_file_path, mode='r', encoding='ISO-8859-1') as file:
    csv_reader = csv.DictReader(file)
    waypoint_count = 1

    for row in csv_reader:
        # Extract latitude, longitude, and altitude directly from CSV
        target_lat = float(row['Lat'])
        target_long = float(row['Lng'])
        altitude = float(row['Alt'])
        velocity = float(row['Spd'])

        # Ensure the velocity is at least 500
        if velocity < 200:
            velocity = 2

        # Convert lat/long to local X, Y coordinates
        x, y = lat_long_to_local_xy_rotated(target_lat, target_long, origin_lat, origin_long, scale)
        z = altitude * scale - offset  # Use altitude as the Z coordinate

        # Construct the waypoint data
        waypoint_data = {
            f"waypoint{waypoint_count}": {
                "x": x,
                "y": y,
                "z": z,
                "velocity": velocity
            }
        }

        # Add the waypoint to the list
        waypoints.append(waypoint_data)
        waypoint_count += 1

# Step 4: Create the final JSON structure and save it
final_data = {
    "Agent1": waypoints
}

json_file_path = 'formatted_data.json'
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(final_data, json_file, indent=4)

print(f"JSON data has been written to {json_file_path}")
