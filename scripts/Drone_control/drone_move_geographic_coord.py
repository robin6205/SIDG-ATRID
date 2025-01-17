import airsim
import time
import json

class DroneController:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.initial_gps_altitude = None

    def initialize_drone(self):
        """Initialize drone and store initial GPS altitude"""
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Store initial GPS altitude
        gps_data = self.client.getGpsData()
        self.initial_gps_altitude = gps_data.gnss.geo_point.altitude
        print(f"Initial GPS altitude: {self.initial_gps_altitude:.2f}m")

    def takeoff(self):
        """Perform takeoff sequence"""
        print("Taking off...")
        self.client.takeoffAsync().join()
        time.sleep(2)  # Stabilization delay

    def move_to_waypoint(self, latitude, longitude, relative_altitude, velocity):
        """Move drone to specified waypoint with relative altitude"""
        target_altitude = self.initial_gps_altitude + relative_altitude
        
        print(f"\nMoving to waypoint:")
        print(f"Latitude: {latitude}")
        print(f"Longitude: {longitude}")
        print(f"Target altitude: {target_altitude:.2f}m (Initial + {relative_altitude}m)")
        print(f"Velocity: {velocity}m/s")
        
        self.client.moveToGPSAsync(
            latitude=latitude,
            longitude=longitude,
            altitude=target_altitude,
            velocity=velocity
        ).join()

    def land_and_reset(self):
        """Perform landing sequence and reset drone"""
        print("\nHovering for 3 seconds before landing...")
        time.sleep(3)
        
        print("Initiating landing sequence...")
        self.client.landAsync().join()
        time.sleep(3)  # Wait for landing to complete
        
        print("Resetting drone...")
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

    def print_current_altitude(self):
        """Print current altitude information"""
        state = self.client.getMultirotorState()
        ned_altitude = -state.kinematics_estimated.position.z_val
        gps = self.client.getGpsData()
        gps_altitude = gps.gnss.geo_point.altitude
        
        print(f"Current NED altitude: {ned_altitude:.2f}m")
        print(f"Current GPS altitude: {gps_altitude:.2f}m")
        print(f"Height above start: {gps_altitude - self.initial_gps_altitude:.2f}m")

def execute_mission(mission_file, target_altitude=30, velocity=10):
    """Execute mission from JSON file"""
    # Load waypoints
    with open(mission_file, 'r') as f:
        mission_data = json.load(f)
    
    # Skip first waypoint (spawn point) and get remaining waypoints
    waypoints = mission_data['waypoints'][1:]
    print(f"Loaded {len(waypoints)} waypoints to navigate")
    
    # Initialize drone controller
    drone = DroneController()
    
    try:
        # Initialize and take off
        drone.initialize_drone()
        drone.takeoff()
        drone.print_current_altitude()
        
        # Navigate through waypoints
        for i, waypoint in enumerate(waypoints, 1):
            print(f"\nNavigating to waypoint {i} of {len(waypoints)}")
            drone.move_to_waypoint(
                latitude=waypoint['latitude'],
                longitude=waypoint['longitude'],
                relative_altitude=target_altitude,
                velocity=velocity
            )
            drone.print_current_altitude()
            time.sleep(1)  # Short pause between waypoints
        
        # Land and cleanup
        drone.land_and_reset()
        print("Mission completed successfully")
        
    except Exception as e:
        print(f"Error during mission: {e}")
        print("Attempting emergency landing...")
        drone.land_and_reset()

def main():
    mission_file = "scripts/Drone_control/mission_data.json"
    execute_mission(
        mission_file,
        target_altitude=30,  # 30 meters above starting altitude
        velocity=10         # 10 m/s
    )

if __name__ == "__main__":
    main()


