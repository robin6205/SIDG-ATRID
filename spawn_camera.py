###################################################
# Spawn_camera.py
# Developer: Joshua Chang
# Date: 4/21/2024
# Description: This script will spawn a camera at specified locations and capture images
# On 'Purdue_airport3_7'
##################################################

from src.utils import lat_long_to_local_xy_rotated, create_vehicle, read_coordinates_from_json
import src.airsim as airsim
import math

agent_config_path = 'test_coordinates/uav_path_test_1.json'
simulation_config_path = 'config/sim_config.json'

sim_config = read_coordinates_from_json(simulation_config_path, 'sim_config')
print(sim_config)
# Reference origin in Unreal Engine (location of Georefernce)
origin_lat = sim_config["origin_latitude"]
origin_long = sim_config["origin_longitude"]
# Assume Unreal units match meters (scale=1)
scale = sim_config["scale"]

# Sensor placement
sensor_location = read_coordinates_from_json(agent_config_path, 'sensors')
for sensor in sensor_location:
    target_lat = sensor[0]
    target_long = sensor[1]
    # if sensor[2] is not provided, set altitude to 1000
    altitdue = sensor[2] if len(sensor) == 3 else 1000
    
    # Convert geographic coordinates to local X, Y coordinates
    x_rot, y_rot = lat_long_to_local_xy_rotated(target_lat, target_long, origin_lat, origin_long, scale)

    # Asset name of the blueprint object in the Unreal project database
    # This should be the name of the asset as seen in Unreal's content browser
    asset_name = "Recording_camera5"

    # Desired name for the new object instance
    object_name = "RecordingCamera5Instance"
    
    z_offset = -65  # Offset where the playerstart is located
    z_adjusted = altitdue + z_offset  # Adjusted z coordinate for Unreal units
    yaw = math.radians(90)

    # Coordinates and orientation for where you want to spawn the object
    x, y, z = x_rot/100, y_rot/100, z_adjusted/100 
    
    # Connect to the AirSim client
    client = airsim.VehicleClient()
    client.confirmConnection()
    
    # Spawn the blueprint object at the specified pose and scale
    spawned_object_name = client.simSpawnObject(object_name=object_name, 
                                                asset_name=asset_name, 
                                                pose=airsim.Pose(airsim.Vector3r(x, y, -z), airsim.to_quaternion(0, 0, yaw)), 
                                                scale=airsim.Vector3r(1, 1, 1),
                                                is_blueprint=True)  
    if spawned_object_name:
        print(f"Successfully spawned '{spawned_object_name}' at the specified location.")
    else:
        print("Failed to spawn the object. Please check the asset name and parameters.")
        break

# spawn uav vehicle at specified location
vehicle_location = read_coordinates_from_json(agent_config_path, 'vehicle')
vehicle_location_lat = vehicle_location[0][0]
vehicle_location_long = vehicle_location[0][1]
vehicle_yaw = math.radians(0)
vehicle_x, vehicle_y = lat_long_to_local_xy_rotated(vehicle_location_lat, vehicle_location_long, origin_lat, origin_long, scale)
vehicle_location = airsim.Vector3r(vehicle_x/100, vehicle_y/100, -z_offset/100)

create_vehicle(client= client, vehicle_name='drone2', location=vehicle_location, rotation=airsim.to_quaternion(0, 0, vehicle_yaw))
