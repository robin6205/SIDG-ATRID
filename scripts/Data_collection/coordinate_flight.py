###################################################
# Coordinate flight json 
# Developer: Joshua Chang
# Date: 5/1/2024
# Description: This script will move the drone to GPS coordinates specified in a JSON file.
# The coordinates are in geographic coordinate system (latitude, longitude) and the drone will move to each location
# On environment 'Purdue_airport3_7'
##################################################
import src.airsim as airsim
from src.utils import read_coordinates_from_json
import json

class DroneController:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.vehicle_name = 'Drone1'
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)

    def move_drone_to_gps_location(self, path_coordinates, altitude, velocity):
        print("Taking off...")
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        gps = self.client.getGpsData(vehicle_name=self.vehicle_name)
        print('Drone taking off')
        print('Current GPS location: ', gps)
        
        current_latitude = gps.gnss.geo_point.latitude
        current_longitude = gps.gnss.geo_point.longitude
        
        print(f"Current GPS location: Latitude={current_latitude}, Longitude={current_longitude}, Altitude={altitude} at {velocity} m/s")
        
        for coord in path_coordinates:
            latitude, longitude = coord['x'], coord['y']
            print(f"Moving to GPS location: Latitude={latitude}, Longitude={longitude}, Altitude={altitude} at {velocity} m/s")
            self.client.moveToGPSAsync(latitude=latitude, longitude=longitude, altitude=altitude, velocity=velocity, vehicle_name=self.vehicle_name).join()

        print("Landing...")
        self.client.landAsync().join()

    def disarm(self):
        # Cleanup by disarming and releasing API control
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

def read_coordinates_from_json(file_path, key):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data[key]

def main():
    controller = DroneController()
    path_coordinates = read_coordinates_from_json('test_coordinates/uav_path_test_5.json', 'path')
    altitude = 350
    velocity = 15  # meters per second

    controller.move_drone_to_gps_location(path_coordinates, altitude, velocity)
    controller.disarm()

if __name__ == '__main__':
    main()
