import src.airsim as airsim
import time

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off to a specified altitude
takeoff_altitude = 350
client.takeoffAsync().join()
client.moveToZAsync(-takeoff_altitude, 5).join()

# Hover for 30 seconds
time.sleep(30)

# Land the drone
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)