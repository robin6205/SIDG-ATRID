import math
import json 
from . import airsim

def read_coordinates_from_json(file_path, key):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            path_coordinates = data.get(key, [])
        return path_coordinates

def list_scene_objects(client):
    
    """
    List all scene objects and assets in the Unreal Engine environment.
    
    Parameters: client (airsim.VehicleClient): AirSim client instance.
    
    returns: None
    """
    # Connect to the AirSim simulator 
    # List all scene objects
    object_names = client.simListSceneObjects()
    asset_names = client.simListAssets()
    print("Available Objects in the Scene:")
    for name in object_names:
        print(name)
        
    print("\nAvailable Assets in the Scene:")
    for name in asset_names:
        print(name)

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

def create_vehicle(client, vehicle_name, location, rotation):
    """
    Create a vehicle at the specified location and rotation using simAddVehicle.
    
    Args:
        client (airsim.VehicleClient): AirSim client instance.
        vehicle_name (str): Name of the vehicle to create.
        location (airsim.Vector3r): Location vector for the vehicle.
        rotation (airsim.Quaternionr): Orientation quaternion for the vehicle.
    
    Returns:
        bool: Whether vehicle was created successfully.
    """
    return client.simAddVehicle(vehicle_name, 'SimpleFlight', airsim.Pose(location, rotation), '')

def lat_long_to_local_xy(target_lat, target_long, origin_lat, origin_long, scale=1):
    """
    Convert latitude and longitude to local X, Y coordinates based on a reference origin.
    
    Parameters:
    - target_lat, target_long: Latitude and longitude of the target point.
    - origin_lat, origin_long: Latitude and longitude of the reference origin point.
    - scale: Scale factor for conversion to Unreal Engine units (default is 1).
    
    Returns:
    - Tuple (x, y) representing local coordinates in Unreal Engine environment.
    """
    # Earth's radius in meters
    R = 6378137
    
    # Convert latitude and longitude differences to radians
    delta_lat = math.radians(target_lat - origin_lat)
    delta_long = math.radians(target_long - origin_long)
    
    # Average latitude for the longitude calculation
    avg_lat = math.radians((origin_lat + target_lat) / 2.0)
    
    # Calculate X, Y distances in meters
    x_distance = R * delta_long * math.cos(avg_lat)
    y_distance = R * delta_lat
    
    # Apply scale if needed
    x = x_distance * scale
    y = y_distance * scale
    
    return x, y