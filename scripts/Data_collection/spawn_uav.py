
from src.utils import lat_long_to_local_xy_rotated, create_vehicle, read_coordinates_from_json
import src.airsim as airsim
import math


agent_config_path = 'test_coordinates/uav_path_test_1.json'
simulation_config_path = 'config/sim_config.json'

sim_config = read_coordinates_from_json(simulation_config_path, 'sim_config')

# Reference origin in Unreal Engine (location of Georefernce)
origin_lat = sim_config["origin_latitude"]
origin_long = sim_config["origin_longitude"]
# Assume Unreal units match meters (scale=1)
scale = sim_config["scale"]
z_offset = -65  # Offset where the playerstart is located
client = airsim.VehicleClient()
client.confirmConnection()
# spawn uav vehicle at specified location
vehicle_location = read_coordinates_from_json(agent_config_path, 'vehicle_spawn')
vehicle_location_lat = vehicle_location[0]["x"]
vehicle_location_long = vehicle_location[0]["y"]
vehicle_yaw = math.radians(0)
vehicle_x, vehicle_y = lat_long_to_local_xy_rotated(vehicle_location_lat, vehicle_location_long, origin_lat, origin_long, scale)
vehicle_location = airsim.Vector3r(vehicle_x / 100, vehicle_y / 100, -z_offset / 100)

create_vehicle(client=client, vehicle_name='drone2', location=vehicle_location, rotation=airsim.to_quaternion(0, 0, vehicle_yaw))
