###################################################
# Coordinate flight json 
# Developer: Joshua Chang
# Date: 5/1/2024
# Description: Description: This script will move the drone to GPS coordinates specified in a JSON file.
# The coordinates are in geographic coordinate system (latitude, longitude) and the drone will move to each location
# On environment 'Purdue_airport3_7'
##################################################
import src.airsim as airsim
from src.utils import read_coordinates_from_json
import json
class DroneController:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.vehicle_name = 'drone2'
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)


    def move_drone_to_gps_location(self, path_coordinates, altitude, velocity):
        print("Taking off...")
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        gps = self.client.getGpsData(vehicle_name=self.vehicle_name)
        print('drone taking off')
        print('current gps location: ', gps)
        
        current_latitude = gps.gnss.geo_point.latitude
        current_longitude = gps.gnss.geo_point.longitude
        
        print(f"Moving to GPS location: Latitude={current_latitude}, Longitude={current_longitude}, Altitude={altitude} at {velocity} m/s")
        
        self.client.moveToGPSAsync(current_latitude, current_longitude, altitude, velocity, vehicle_name=self.vehicle_name).join()

        for coord in path_coordinates:
            latitude, longitude = coord
            print(f"Moving to GPS location: Latitude={latitude}, Longitude={longitude}, Altitude={altitude} at {velocity} m/s")
            self.client.moveToGPSAsync(latitude=latitude, longitude=longitude, altitude=altitude, velocity=velocity, vehicle_name=self.vehicle_name).join()

        print("Landing...")
        self.client.landAsync().join()

    def disarm(self):
        # Cleanup by disarming and releasing API control
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

def main():
    controller = DroneController()
    path_coordinates = read_coordinates_from_json('test_coordinates/uav_path_test_1.json', 'path')
    altitude = 150
    velocity = 15  # meters per second

    controller.move_drone_to_gps_location(path_coordinates, altitude, velocity)
    controller.disarm()

if __name__ == '__main__':
    main()
