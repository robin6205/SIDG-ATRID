import json
import time
import signal
import sys
import os
import re
import traceback
import argparse
import math
from datetime import datetime
from unrealcv import Client
from airsim import MultirotorClient
import threading

class ParallelDataCollection:
    def __init__(self, config_file, width=None, height=None, save_state=False):
        # Store resolution settings and state saving flag
        self.width = width
        self.height = height
        self.save_state = save_state
        
        # Load initial configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components that persist across runs
        self.unrealcv_client = Client(('127.0.0.1', 9000))
        self.airsim_client = None # Initialize to None, connection will be handled in _setup_run_parameters
        self.shutting_down = False  # Add shutdown flag
        self.continue_capture = True  # Flag to control capture loop
        self.agent_list_path = "D:/Unreal Projects/ACREDataCollection/AgentList.json" # This path is constant
        self.camera_dirs = {} # Initialize camera_dirs here
        
        # Connect to UnrealCV
        print("Connecting to UnrealCV...")
        self.unrealcv_client.connect()
        if not self.unrealcv_client.isconnected():
            raise ConnectionError("Could not connect to UnrealCV server")
        print("Successfully connected to UnrealCV")
        
        # Set resolution if provided (UnrealCV client is already connected)
        if self.width is not None and self.height is not None:
            print(f"Setting resolution to {self.width}x{self.height}")
            self.unrealcv_client.request(f'r.setres {self.width}x{self.height}')
        
        # Register signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

    def _connect_airsim(self):
        self.airsim_client = None
        try:
            print("Connecting to AirSim for current level...")
            self.airsim_client = MultirotorClient()
            max_retries = 10
            retry_delay = 3  # seconds
            for attempt in range(max_retries):
                try:
                    print(f"AirSim connection attempt {attempt+1}/{max_retries}")
                    self.airsim_client.confirmConnection()
                    print("Successfully connected to AirSim for current level")
                    break
                except Exception as e:
                    print(f"Failed to connect to AirSim for current level: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("AirSim connection failed for current level. Will run in capture-only mode.")
                        self.airsim_client = None
        except Exception as e:
            print(f"Error setting up AirSim for current level: {e}")
            self.airsim_client = None

    def _setup_run_parameters(self, config_file):
        # Load configuration for the current run
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Initialize drone type in AirSim settings (this needs to happen before AirSim connection)
        drone_type = self.config['drone_config'].get('drone_type', 'DJIS900')
        self.init_drone_type(drone_type)
        
        # Initialize data collection parameters based on the new config
        base_dir = self.config['data_collection']['base_output_dir']
        level = self.config['data_collection']['level']
        
        # This will be updated for each time_of_day and focal_length iteration
        self.current_time_of_day_float = None 
        self.current_time_of_day_name = None
        self.current_focal_length = None
        # New structure: base_output_dir/drone_type/level/
        self.level_output_dir = os.path.join(base_dir, drone_type, level)
        print(f"Output directory structure: {base_dir}/{drone_type}/{level}/...")
        
        self.frame_rate = self.config['data_collection']['frame_rate']
        self.capture_duration = self.config['data_collection']['capture_duration']
        self.max_images = self.config['data_collection'].get('max_images', float('inf'))
        self.check_exclusive_mask_color = self.config['data_collection'].get('check_exclusive_mask_color', False)
        
        # Get focal length settings (default to empty list if not specified)
        self.focal_lengths = self.config['data_collection'].get('focal_length', [])
        
    def _prepare_output_directories(self, time_of_day_float, focal_length=None):
        # Update current time of day and focal length
        self.current_time_of_day_float = time_of_day_float
        self.current_time_of_day_name = str(time_of_day_float)
        self.current_focal_length = focal_length
        
        # Create output directories for each camera under the current time of day and focal length
        self.camera_dirs = {}
        for camera_id in self.config['camera_config'].keys():
            camera_num = camera_id.replace('camera', '')
            
            # New structure: base_output_dir/drone_type/level/time_of_day_name/focal_length/camera_#/
            if focal_length is not None:
                camera_base_dir = os.path.join(self.level_output_dir, self.current_time_of_day_name, f"focal_{focal_length}", camera_id)
            else:
                camera_base_dir = os.path.join(self.level_output_dir, self.current_time_of_day_name, camera_id)
            
            # Set up directories for this camera
            self.camera_dirs[camera_id] = {
                'rgb': os.path.join(camera_base_dir, 'rgb'),
                'mask': os.path.join(camera_base_dir, 'mask')
            }
            
            # Add state directory if state saving is enabled
            if self.save_state:
                self.camera_dirs[camera_id]['state'] = os.path.join(camera_base_dir, 'state')
                os.makedirs(self.camera_dirs[camera_id]['state'], exist_ok=True)
                print(f"Created state directory for {camera_id}: {self.camera_dirs[camera_id]['state']}")
            
            # Create directories
            os.makedirs(self.camera_dirs[camera_id]['rgb'], exist_ok=True)
            os.makedirs(self.camera_dirs[camera_id]['mask'], exist_ok=True)
            
            # Create agent color info file specific to this camera
            self.camera_dirs[camera_id]['agent_color_info'] = os.path.join(
                camera_base_dir, "agent_color_info.json"
            )

        # Get the starting frame index based on existing files in the current directory
        self.frame_index = self._get_max_frame_index()
        focal_info = f" focal_{focal_length}" if focal_length is not None else ""
        print(f"Starting data collection from frame index: {self.frame_index} for {self.current_time_of_day_name}{focal_info} in {self.config['data_collection']['level']}")

    def _signal_handler(self, sig, frame):
        """Handle interrupt signal for immediate shutdown"""
        print('\nImmediate shutdown requested...')
        self.continue_capture = False
        time.sleep(1)  # Give a moment for threads to clean up
        os._exit(0)  # Force immediate exit

    def _get_max_frame_index(self):
        """Find the highest frame index from existing files across all cameras."""
        max_index = -1
        
        for camera_dirs in self.camera_dirs.values():
            for directory in [camera_dirs['rgb'], camera_dirs['mask']]:
                if not os.path.exists(directory):
                    continue
                    
                for filename in os.listdir(directory):
                    match = re.match(r'^(\d+)_\d{8}_\d{6}_', filename)
                    if match:
                        index = int(match.group(1))
                        max_index = max(max_index, index)
        
        return max_index + 1 if max_index >= 0 else 0

    def load_agent_list(self):
        """Load the agent list from JSON."""
        with open(self.agent_list_path, 'r') as agent_file:
            return json.load(agent_file)

    def get_agent_colors(self, agent_list):
        """Retrieve colors of all agents and save them to JSON files for each camera."""
        color_str = self.unrealcv_client.request('vget /object/Drone1/color')
        match = re.match(r'\(R=(\d+),G=(\d+),B=(\d+),A=(\d+)\)', color_str)
        if match:
            color_data = {
                'R': int(match.group(1)),
                'G': int(match.group(2)),
                'B': int(match.group(3)),
                'A': int(match.group(4))
            }
            print(f"Drone1 color: {color_data}")
        else:
            print("Failed to retrieve color for Drone1")
        agent_color_data = {'Drone1': color_data}
        # Save color info for each camera
        for camera_id, dirs in self.camera_dirs.items():
            with open(dirs['agent_color_info'], 'w') as json_file:
                json.dump(agent_color_data, json_file, indent=4)
            print(f"Saved agent color data for {camera_id} to {dirs['agent_color_info']}")
        return agent_color_data

    def get_drone_color_by_type(self, drone_type):
        """Get the appropriate color for the drone based on its type."""
        if drone_type == "Opterra":
            return (255, 255, 0)  # Blue color for Opterra
        elif drone_type in ["DJIS900", "DJI_S900", "DJIS-900"]:
            return (255, 255, 0)  # Yellow color for DJI S900 variants
        else:
            # Default to yellow for unknown drone types
            print(f"Unknown drone type '{drone_type}', defaulting to yellow")
            return (255, 255, 0)

    def set_agent_color(self, agent_list, target_agent_name, drone_type):
        """Set a specific color for a target agent based on drone type and handle conflicting colors"""
        # Get the target color based on drone type
        target_color = self.get_drone_color_by_type(drone_type)
        color_name = "red" if target_color == (255, 0, 0) else "yellow"
        
        # Pause simulation
        self.unrealcv_client.request('vset /action/game/pause')
        objects_response = self.unrealcv_client.request('vget /objects')
        all_objects = objects_response.split(' ')
        print(f"Found {len(all_objects)} objects in the scene")
        
        # Check each object's color and change if it matches our target color
        for obj_name in all_objects:
            if not obj_name or obj_name == 'Drone1':  # Skip empty strings and the drone
                continue
            try:
                # Get current color of the object
                color_str = self.unrealcv_client.request(f'vget /object/{obj_name}/color')
                match = re.match(r'\(R=(\d+),G=(\d+),B=(\d+),A=(\d+)\)', color_str)
                if match:
                    current_color = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                    # If the object has the same color as our target drone color, change it
                    if current_color == target_color:
                        print(f"Found conflicting {color_name} color on object: {obj_name}")
                        # Set to blue (0, 0, 255) as alternative color
                        self.unrealcv_client.request(f'vset /object/{obj_name}/color 0 0 255')
                        print(f"Changed color of {obj_name} to blue")
                        break  # Exit after handling the first conflict
            except Exception as e:
                print(f"Error processing object {obj_name}: {e}")
                continue
        
        # Finally, set the drone to the target color
        r, g, b = target_color
        self.unrealcv_client.request(f'vset /object/Drone1/color {r} {g} {b}')
        print(f"Set Drone1 color to {color_name} ({r}, {g}, {b}) for drone type {drone_type}")
        
        # Resume simulation
        self.unrealcv_client.request('vset /action/game/resume')

    def setup_all_cameras(self):
        """Set up all cameras at once."""
        print("Setting up all cameras...")
        
        # Ensure simulation is running for camera setup
        if self.unrealcv_client.request('vget /action/game/is_paused') == 'true':
            self.unrealcv_client.request('vset /action/game/resume')
            print("Resumed simulation for camera setup")
        
        # Debug: Check available cameras before setup
        camera_list_before = self.unrealcv_client.request('vget /cameras')
        print(f"Available cameras before setup: {camera_list_before}")
        
        # Set up each camera
        for camera_id, camera_config in self.config['camera_config'].items():
            # Convert camera_id to numeric ID (e.g., 'camera1' -> 1)
            camera_num = int(camera_id.replace('camera', ''))
            
            print(f"\n=== Setting up {camera_id} (camera {camera_num}) ===")
            
            # Spawn and configure the camera
            print(f"Spawning camera {camera_num}...")
            spawn_result1 = self.unrealcv_client.request('vset /cameras/spawn')
            spawn_result2 = self.unrealcv_client.request('vset /cameras/spawn')
            print(f"Spawn results for camera {camera_num}: {spawn_result1}, {spawn_result2}")
            
            # Wait a moment for camera to be fully spawned
            time.sleep(0.5)
            
            # Verify camera exists
            camera_list_current = self.unrealcv_client.request('vget /cameras')
            print(f"Available cameras after spawn: {camera_list_current}")
            
            attached_to_drone = camera_config.get('attached_to_drone', False)

            if attached_to_drone:
                self._attach_camera_to_drone(camera_num, camera_config)
                print(f"=== {camera_id} setup complete (attached to drone) ===")
                print(f"Relative offset: {camera_config.get('relative_offset', {})}")
                print(f"Relative rotation: {camera_config.get('rotation', {})}")
                print()
            else:
                location = camera_config.get("location")
                if not location:
                    print(f"Warning: {camera_id} missing 'location' for static setup. Skipping position set.")
                else:
                    # Set camera location
                    location_cmd = (
                        f"vset /camera/{camera_num}/location "
                        f"{location['x']} {location['y']} {location['z']}"
                    )
                    location_result = self.unrealcv_client.request(location_cmd)
                    print(f"Location command: {location_cmd}")
                    print(f"Location result: {location_result}")

                    # Verify location was set
                    verify_location = self.unrealcv_client.request(f'vget /camera/{camera_num}/location')
                    print(f"Verified location: {verify_location}")

                # Set camera rotation (format numbers to avoid scientific notation)
                rotation = camera_config.get("rotation", {})
                pitch = float(rotation.get("pitch", 0.0))
                yaw = float(rotation.get("yaw", 0.0))
                roll = float(rotation.get("roll", 0.0))
                rotation_cmd = f'vset /camera/{camera_num}/rotation {pitch:.6f} {yaw:.6f} {roll:.6f}'
                rotation_result = self.unrealcv_client.request(rotation_cmd)
                print(f"Rotation command: {rotation_cmd}")
                print(f"Rotation result: {rotation_result}")

                # Verify rotation was set
                verify_rotation = self.unrealcv_client.request(f'vget /camera/{camera_num}/rotation')
                print(f"Verified rotation: {verify_rotation}")

                # Also try setting rotation again if it didn't work the first time
                if verify_rotation == "0.000000 0.000000 0.000000" or not verify_rotation or verify_rotation == 'error':
                    print(f"Rotation not applied correctly, trying again...")
                    time.sleep(0.2)
                    # Use the same formatted command
                    rotation_result2 = self.unrealcv_client.request(rotation_cmd)
                    print(f"Second rotation attempt result: {rotation_result2}")
                    verify_rotation2 = self.unrealcv_client.request(f'vget /camera/{camera_num}/rotation')
                    print(f"Verified rotation after second attempt: {verify_rotation2}")

                print(f"=== {camera_id} setup complete ===")
                if location:
                    print(f"Target location: {location}")
                    print(f"Actual location: {verify_location}")
                print(f"Target rotation: {rotation}")
                print(f"Actual rotation: {verify_rotation}")
                print()
        
        # Debug: Check available cameras after setup
        camera_list_after = self.unrealcv_client.request('vget /cameras')
        print(f"Available cameras after setup: {camera_list_after}")
        print("All cameras set up successfully")

    def _attach_camera_to_drone(self, camera_num, camera_config):
        """Attach a spawned camera to the drone using a relative offset and fixed rotation."""
        offset = camera_config.get('relative_offset')
        if offset is None:
            print(f"Warning: camera {camera_num} missing 'relative_offset' for attachment. Skipping attach command.")
            return

        rotation = camera_config.get('rotation', {})
        pitch = float(rotation.get('pitch', 0.0))
        yaw = float(rotation.get('yaw', 0.0))
        roll = float(rotation.get('roll', 0.0))

        attach_cmd = (
            f"vrun ce cameraattach {camera_num} "
            f"{float(offset.get('x', 0.0)):.6f} "
            f"{float(offset.get('y', 0.0)):.6f} "
            f"{float(offset.get('z', 0.0)):.6f} "
            f"{pitch:.6f} {yaw:.6f} {roll:.6f}"
        )

        # TODO: Update the Unreal Engine blueprint handler to process `cameraattach` and reparent the camera to Drone1.

        try:
            print(f"Attach command: {attach_cmd}")
            attach_result = self.unrealcv_client.request(attach_cmd)
            print(f"Attach result: {attach_result}")
        except Exception as e:
            print(f"Error sending attach command for camera {camera_num}: {e}")

    def _initialize_drone(self):
        """Initialize drone and store initial position"""
        if self.airsim_client is None:
            print("AirSim not available. Skipping drone initialization.")
            return False
        # Reset and clean state
        self.airsim_client.enableApiControl(True)
        # self.airsim_client.reset()
        self.airsim_client.enableApiControl(True)
        self.airsim_client.armDisarm(False)
        self.airsim_client.armDisarm(True)
        # Store initial position for return to home
        state = self.airsim_client.getMultirotorState()
        self.initial_position = state.kinematics_estimated.position
        self.initial_z = self.initial_position.z_val
        print(f"Initial position: X={self.initial_position.x_val:.2f}m, Y={self.initial_position.y_val:.2f}m, Z={self.initial_z:.2f}m")
        # Store initial GPS altitude for geographic coordinates
        gps_data = self.airsim_client.getGpsData()
        self.initial_gps_altitude = gps_data.gnss.geo_point.altitude
        print(f"Initial GPS altitude: {self.initial_gps_altitude:.2f}m")
        return True

    def _takeoff(self):
        """Perform takeoff sequence"""
        if self.airsim_client is None:
            print("AirSim not available. Skipping takeoff.")
            return
            
        print("Taking off...")
        self.airsim_client.takeoffAsync().join()
        time.sleep(2)  # Stabilization delay

    def _move_to_waypoint(self, x, y, relative_altitude, velocity):
        """Move drone to specified waypoint with relative altitude"""
        if self.airsim_client is None:
            print("AirSim not available. Skipping waypoint movement.")
            return
            
        z = -relative_altitude
        
        # Get current position to calculate yaw
        current_pose = self.airsim_client.simGetVehiclePose()
        current_x = current_pose.position.x_val
        current_y = current_pose.position.y_val
        
        # Calculate yaw angle to face the waypoint
        target_yaw = self._calculate_yaw_to_waypoint(current_x, current_y, x, y)
        
        print(f"\nMoving to waypoint:")
        print(f"X: {x}m")
        print(f"Y: {y}m")
        print(f"Relative altitude: {relative_altitude}m")
        print(f"Velocity: {velocity}m/s")
        print(f"Rotating to face waypoint (yaw: {target_yaw:.1f}°)")
        
        # First rotate to face the waypoint
        self.airsim_client.rotateToYawAsync(target_yaw).join()
        
        # Then move to the waypoint
        self.airsim_client.moveToPositionAsync(
            x=x,
            y=y,
            z=z,
            velocity=velocity
        ).join()

    def _calculate_yaw_to_waypoint(self, current_x, current_y, target_x, target_y):
        """Calculate the yaw angle (in degrees) needed to face the target waypoint"""
        # Calculate the direction vector
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Calculate the yaw angle in radians, then convert to degrees
        # atan2 gives angle from positive x-axis, counterclockwise
        # AirSim uses NED coordinate system where positive yaw is clockwise from North (positive Y)
        yaw_radians = math.atan2(dx, dy)  # Note: dx, dy order for NED system
        yaw_degrees = math.degrees(yaw_radians)
        
        return yaw_degrees

    def _calculate_yaw_to_geographic_waypoint(self, current_lat, current_lon, target_lat, target_lon):
        """Calculate the yaw angle (in degrees) needed to face the target geographic waypoint"""
        # Convert lat/lon differences to approximate local coordinates
        # This is an approximation for small distances
        dlat = target_lat - current_lat
        dlon = target_lon - current_lon
        
        # Calculate bearing (yaw) - this is the angle from North
        # atan2(dlon, dlat) gives the bearing from current to target position
        bearing_radians = math.atan2(dlon, dlat)
        bearing_degrees = math.degrees(bearing_radians)
        
        return bearing_degrees

    def _move_to_geographic_waypoint(self, latitude, longitude, relative_altitude, velocity):
        """Move drone to specified waypoint using geographic coordinates"""
        if self.airsim_client is None:
            print("AirSim not available. Skipping geographic waypoint movement.")
            return
            
        target_altitude = self.initial_gps_altitude + relative_altitude
        
        # Get current GPS position to calculate yaw
        current_gps = self.airsim_client.getGpsData()
        current_lat = current_gps.gnss.geo_point.latitude
        current_lon = current_gps.gnss.geo_point.longitude
        
        # Calculate yaw angle to face the waypoint
        target_yaw = self._calculate_yaw_to_geographic_waypoint(current_lat, current_lon, latitude, longitude)
        
        print(f"\nMoving to waypoint:")
        print(f"Latitude: {latitude}")
        print(f"Longitude: {longitude}")
        print(f"Target altitude: {target_altitude:.2f}m (Initial + {relative_altitude}m)")
        print(f"Velocity: {velocity}m/s")
        print(f"Rotating to face waypoint (yaw: {target_yaw:.1f}°)")
        
        # First rotate to face the waypoint
        self.airsim_client.rotateToYawAsync(target_yaw).join()
        
        # Then move to the waypoint
        self.airsim_client.moveToGPSAsync(
            latitude=latitude,
            longitude=longitude,
            altitude=target_altitude,
            velocity=velocity
        ).join()

    def _land_and_reset(self):
        """Perform landing sequence and reset drone"""
        if self.airsim_client is None:
            print("AirSim not available. Skipping landing and reset.")
            return
            
        print("\nHovering for 3 seconds before landing...")
        time.sleep(3)
        
        print("Initiating return to home for 5 seconds...")
        # Attempt to return home for a limited time
        self.airsim_client.goHomeAsync()
        time.sleep(5) # Allow 5 seconds for it to go home
        print("Return to home attempt finished (5 seconds elapsed). Proceeding to reset.")
        
        try:
            print("Resetting drone...")
            self.airsim_client.reset()
            print("Drone reset complete")
        except Exception as e:
            print(f"Error resetting drone: {e}")

    def _print_current_position(self):
        """Print current position information"""
        if self.airsim_client is None:
            print("AirSim not available. Cannot print position.")
            return
            
        state = self.airsim_client.getMultirotorState()
        position = state.kinematics_estimated.position
        
        print(f"Current position:")
        print(f"X: {position.x_val:.2f}m")
        print(f"Y: {position.y_val:.2f}m")
        print(f"Z: {position.z_val:.2f}m")
        print(f"Height above start: {-(position.z_val - self.initial_z):.2f}m")

    def _print_current_altitude(self):
        """Print current altitude information"""
        if self.airsim_client is None:
            print("AirSim not available. Cannot print altitude.")
            return
            
        state = self.airsim_client.getMultirotorState()
        ned_altitude = -state.kinematics_estimated.position.z_val
        gps = self.airsim_client.getGpsData()
        gps_altitude = gps.gnss.geo_point.altitude
        
        print(f"Current NED altitude: {ned_altitude:.2f}m")
        print(f"Current GPS altitude: {gps_altitude:.2f}m")
        print(f"Height above start: {gps_altitude - self.initial_gps_altitude:.2f}m")

    def _get_ground_truth_data(self, camera_id, timestamp, frame_index):
        """
        Get ground truth position and rotation data for drone and camera.
        Returns a dictionary with drone and camera state information.
        """
        state_data = {
            'timestamp': timestamp,
            'frame_index': frame_index,
            'drone': {},
            'camera': {}
        }
        
        # Get drone data from UnrealCV (consistent with camera coordinate system)
        try:
            # Get drone position using UnrealCV - should be "Drone1" based on your comment
            drone_pos_str = self.unrealcv_client.request('vget /object/Drone1/location')
            drone_rot_str = self.unrealcv_client.request('vget /object/Drone1/rotation')
            
            # Parse drone position (format: "X Y Z")
            if drone_pos_str and drone_pos_str != 'error':
                pos_parts = drone_pos_str.split()
                drone_position = {
                    'x': float(pos_parts[0]),
                    'y': float(pos_parts[1]),
                    'z': float(pos_parts[2])
                }
            else:
                drone_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            
            # Parse drone rotation (format: "Pitch Yaw Roll")
            if drone_rot_str and drone_rot_str != 'error':
                rot_parts = drone_rot_str.split()
                drone_rotation = {
                    'pitch': float(rot_parts[0]),
                    'yaw': float(rot_parts[1]),
                    'roll': float(rot_parts[2])
                }
            else:
                drone_rotation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            
            state_data['drone'] = {
                'position': drone_position,
                'rotation': drone_rotation
            }
            
        except Exception as e:
            print(f"Error getting drone state from UnrealCV: {e}")
            state_data['drone'] = {
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'rotation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                'error': str(e)
            }
        
        # Get camera data from UnrealCV
        try:
            camera_num = int(camera_id.replace('camera', ''))
            
            # Get camera position using UnrealCV
            camera_pos_str = self.unrealcv_client.request(f'vget /camera/{camera_num}/location')
            camera_rot_str = self.unrealcv_client.request(f'vget /camera/{camera_num}/rotation')
            
            # Parse position (format: "X Y Z")
            if camera_pos_str and camera_pos_str != 'error':
                pos_parts = camera_pos_str.split()
                camera_position = {
                    'x': float(pos_parts[0]),
                    'y': float(pos_parts[1]),
                    'z': float(pos_parts[2])
                }
            else:
                camera_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            
            # Parse rotation (format: "Pitch Yaw Roll")
            if camera_rot_str and camera_rot_str != 'error':
                rot_parts = camera_rot_str.split()
                camera_rotation = {
                    'pitch': float(rot_parts[0]),
                    'yaw': float(rot_parts[1]),
                    'roll': float(rot_parts[2])
                }
            else:
                camera_rotation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            
            state_data['camera'] = {
                'id': camera_id,
                'camera_number': camera_num,
                'position': camera_position,
                'rotation': camera_rotation
            }
            
        except Exception as e:
            print(f"Error getting camera state from UnrealCV: {e}")
            state_data['camera'] = {
                'id': camera_id,
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'rotation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                'error': str(e)
            }
        
        return state_data

    def _save_state_to_file(self, camera_id, timestamp, frame_index):
        """
        Save ground truth state data to a JSON file.
        Creates a file with the same naming pattern as the images.
        """
        if not self.save_state:
            return
            
        try:
            # Get ground truth data
            state_data = self._get_ground_truth_data(camera_id, timestamp, frame_index)
            
            # Create filename matching the image pattern (but with .json extension)
            state_filename = f"{frame_index}_{timestamp}_{camera_id}_state.json"
            state_path = os.path.join(self.camera_dirs[camera_id]['state'], state_filename)
            
            # Create structured JSON data
            json_data = {
                "metadata": {
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "coordinate_system": "unreal_units"
                },
                "agents": {
                    "Drone1": {
                        "position": state_data['drone']['position'],
                        "rotation": state_data['drone']['rotation']
                    },
                    camera_id: {
                        "position": state_data['camera']['position'],
                        "rotation": state_data['camera']['rotation'],
                        "camera_number": state_data['camera'].get('camera_number', None)
                    }
                }
            }
            
            # Add error information if present
            if 'error' in state_data['drone']:
                json_data['entities']['Drone1']['error'] = state_data['drone']['error']
            if 'error' in state_data['camera']:
                json_data['entities'][camera_id]['error'] = state_data['camera']['error']
            
            # Write state data to JSON file
            with open(state_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"Saved state data to: {state_path}")
            
        except Exception as e:
            print(f"Error saving state data for {camera_id}: {e}")

    def capture_frames_parallel(self):
        """Capture RGB and object mask frames from all cameras in parallel."""
        start_time = time.time()
        
        print(f"Starting parallel capture for all cameras, max images: {self.max_images}")
        
        # Main capture loop
        while self.continue_capture and self.frame_index < self.max_images and (time.time() - start_time) < self.capture_duration:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Pause simulation
            self.unrealcv_client.request('vset /action/game/pause')
            
            # Capture images from all cameras
            for camera_id in self.config['camera_config'].keys():
                camera_num = int(camera_id.replace('camera', ''))
                
                # Capture and save object mask image
                mask_filename = f"{self.frame_index}_{timestamp}_{camera_id}_object_mask.png"
                mask_path = os.path.join(self.camera_dirs[camera_id]['mask'], mask_filename)
                self.unrealcv_client.request(f'vget /camera/{camera_num}/object_mask {mask_path}')
                
                # Capture and save RGB image
                rgb_filename = f"{self.frame_index}_{timestamp}_{camera_id}_lit.png"
                rgb_path = os.path.join(self.camera_dirs[camera_id]['rgb'], rgb_filename)
                self.unrealcv_client.request(f'vget /camera/{camera_num}/lit {rgb_path}')
                
                # Save state data if enabled
                if self.save_state:
                    self._save_state_to_file(camera_id, timestamp, self.frame_index)
            
            # Print status occasionally
            if self.frame_index % 50 == 0:
                print(f"Captured frame {self.frame_index} for all cameras")
                if self.save_state:
                    print(f"  State saving is ENABLED")
                else:
                    print(f"  State saving is DISABLED")
            
            self.frame_index += 1
            
            # Resume simulation
            self.unrealcv_client.request('vset /action/game/resume')
        
        # Determine why we stopped
        if self.frame_index >= self.max_images:
            print(f"Reached maximum frame count of {self.max_images}")
        elif not self.continue_capture:
            print("Data collection stopped because mission completed")
        else:
            print(f"Reached capture duration limit of {self.capture_duration} seconds")
        
        # Print summary
        print(f"Captured {self.frame_index} frames for all cameras")
        total_files = self.frame_index * len(self.camera_dirs) * (3 if self.save_state else 2)  # RGB, mask, and optionally state
        print(f"Total files captured: {total_files}")
        for camera_id, dirs in self.camera_dirs.items():
            print(f"{camera_id} files saved to:")
            print(f"  RGB: {dirs['rgb']} ({self.frame_index} images)")
            print(f"  Mask: {dirs['mask']} ({self.frame_index} images)")
            if self.save_state:
                print(f"  State: {dirs['state']} ({self.frame_index} JSON files)")

    def _execute_mission(self):
        """Execute the drone mission based on the coordinate system"""
        if self.airsim_client is None:
            print("AirSim not available. Cannot execute mission.")
            return
            
        mission_file = self.config['drone_config']['mission_file']
        
        with open(mission_file, 'r') as f:
            mission_data = json.load(f)
        
        waypoints = mission_data['drones'][0]['waypoints']
        
        # Print initial position
        if self.config['drone_config']['coordinate_system'] == 'geographic':
            self._print_current_altitude()
        else:
            self._print_current_position()
        
        # Navigate through waypoints
        for i, waypoint in enumerate(waypoints):
            print(f"\nNavigating to waypoint {waypoint['number']} of {len(waypoints)}")
            
            # Use waypoint speed if specified, otherwise default to 5 m/s
            velocity = waypoint['speed'] if waypoint['speed'] is not None else 5
            
            if self.config['drone_config']['coordinate_system'] == 'geographic':
                self._move_to_geographic_waypoint(
                    latitude=waypoint['latitude'],
                    longitude=waypoint['longitude'],
                    relative_altitude=waypoint['altitude'],
                    velocity=velocity
                )
                self._print_current_altitude()
            else:
                self._move_to_waypoint(
                    x=waypoint['x'],
                    y=waypoint['y'],
                    relative_altitude=waypoint['altitude'],
                    velocity=velocity
                )
                self._print_current_position()
            
            time.sleep(1)  # Short pause between waypoints

    def _set_weather(self, time_value):
        """Set the weather (time of day) in UnrealCV."""
        print(f"Setting weather to time of day: {time_value}")
        weather_cmd = f"vrun ce weatherset {time_value}"
        result = self.unrealcv_client.request(weather_cmd)
        print(f"Weather set result: {result}")
        time.sleep(2) # Give a moment for weather to apply

    def _set_focal_length(self, focal_length):
        """Set the focal length for all cameras."""
        print(f"Setting focal length to: {focal_length}cm")
        focal_range = 200.0  # Fixed focal range
        
        for camera_id in self.config['camera_config'].keys():
            camera_num = int(camera_id.replace('camera', ''))
            focal_cmd = f'vset /camera/{camera_num}/focal {focal_length} {focal_range}'
            result = self.unrealcv_client.request(focal_cmd)
            print(f"Camera {camera_num} focal length set: {result}")
        
        time.sleep(1)  # Give a moment for settings to apply
        print(f"All cameras set to focal length: {focal_length}cm, range: {focal_range}cm")

    def _change_level(self, level_name):
        """Change the UnrealCV level and wait for it to load."""
        print(f"Changing level to: {level_name}")
        level_cmd = f"vset /action/game/level {level_name}"
        result = self.unrealcv_client.request(level_cmd)
        print(f"Level change result: {result}")
        print("Waiting 20 seconds for level to load...")
        time.sleep(20) # Increased wait for level to fully load
        print(f"Level {level_name} loaded.")

    def _stop_capture(self):
        """Stop frame capture and ensure simulation is running"""
        self.continue_capture = False
        # Ensure simulation is running
        self.unrealcv_client.request('vset /action/game/resume')

    def run(self, config_files):
        """Run the parallel data collection process for multiple levels and times of day"""
        for i, config_file in enumerate(config_files):
            print(f"\nProcessing configuration file: {config_file}")
            # Load the configuration for the current file and (re)connect AirSim
            self._setup_run_parameters(config_file) 
            current_level = self.config['data_collection']['level']
            if i > 0: # If it's not the very first configuration file, change the level
                self._change_level(current_level)
            self._connect_airsim()
            # Setup cameras and agent colors for this level
            self.setup_all_cameras()
            agent_list = self.load_agent_list()
            drone_type = self.config['drone_config']['drone_type']
            self.set_agent_color(agent_list, "Drone1", drone_type)
            agent_color_data = self.get_agent_colors(agent_list)
            # Check if AirSim is available for drone control
            airsim_available = self.airsim_client is not None
            time_of_day_values = self.config['data_collection']['time_of_day']
            
            for time_value in time_of_day_values:
                print(f"\n--- Starting data collection for {current_level} at time {time_value} ---")
                self._set_weather(time_value)
                
                # Handle focal length iterations
                if self.focal_lengths:  # If focal lengths are specified
                    for focal_length in self.focal_lengths:
                        print(f"\n--- Setting focal length {focal_length}cm for time {time_value} ---")
                        self.continue_capture = True # Reset capture flag for each new focal length
                        self._prepare_output_directories(time_value, focal_length)
                        # Get and save agent colors
                        self.get_agent_colors(agent_list)
                        # Set focal length for all cameras
                        self._set_focal_length(focal_length)
                        
                        if airsim_available:
                            print("Running in full drone + parallel capture mode")
                            self._initialize_drone()
                            self._takeoff()
                            # Start capture thread
                            capture_thread = threading.Thread(target=self.capture_frames_parallel)
                            capture_thread.start()
                            # Execute mission
                            self._execute_mission()
                            # Stop capture
                            self._stop_capture()
                            # Wait for capture thread to finish
                            if capture_thread.is_alive():
                                capture_thread.join(timeout=10)
                            # Land and reset after each focal length
                            self._land_and_reset()
                        else:
                            print("AirSim not available. Running in capture-only mode.")
                            # Just run the capture function directly
                            self.capture_frames_parallel()
                        print(f"--- Completed data collection for focal length {focal_length}cm ---")
                        time.sleep(3) # Short pause before next focal length
                else:  # No focal lengths specified, use default
                    print(f"\n--- Using default focal length for time {time_value} ---")
                    self.continue_capture = True # Reset capture flag for each new time of day
                    self._prepare_output_directories(time_value)
                    # Get and save agent colors
                    self.get_agent_colors(agent_list)
                    
                    if airsim_available:
                        print("Running in full drone + parallel capture mode")
                        self._initialize_drone()
                        self._takeoff()
                        # Start capture thread
                        capture_thread = threading.Thread(target=self.capture_frames_parallel)
                        capture_thread.start()
                        # Execute mission
                        self._execute_mission()
                        # Stop capture
                        self._stop_capture()
                        # Wait for capture thread to finish
                        if capture_thread.is_alive():
                            capture_thread.join(timeout=10)
                        # Land and reset after each time of day
                        self._land_and_reset()
                    else:
                        print("AirSim not available. Running in capture-only mode.")
                        # Just run the capture function directly
                        self.capture_frames_parallel()
                
                print(f"--- Completed data collection for {current_level} at time {time_value} ---")
                time.sleep(5) # Short pause before next time of day
            print(f"Completed all time of day scenarios for {current_level}.")
        print("\nAll levels and time of day scenarios completed.")
        self.unrealcv_client.disconnect()
        if self.airsim_client: # Disconnect AirSim client if it was connected
            try:
                self.airsim_client.enableApiControl(False)
                self.airsim_client.armDisarm(False)
                # No explicit disconnect() method is needed or available in some AirSim versions.
                # Setting self.airsim_client to None is usually sufficient for cleanup.
                print("Final AirSim client cleanup complete.")
            except Exception as e:
                print(f"Error during final AirSim client cleanup: {e}")
        print("Mission completed and clients disconnected.")

    def init_drone_type(self, drone_type):
        """Update the AirSim settings.json file to use the specified drone type."""
        print(f"Initializing drone type: {drone_type}")
        # Path to AirSim settings.json
        settings_path = os.path.expanduser("~/Documents/AirSim/settings.json")
        pawn_bp = f"Class '/AirSim/Blueprints/BP_{drone_type}.BP_{drone_type}_C'"
        try:
            if not os.path.exists(settings_path):
                print(f"AirSim settings file not found at {settings_path}. Creating new one.")
                settings = {
                    "SettingsVersion": 1.2,
                    "SimMode": "Multirotor",
                    "PawnPaths": {
                        "DefaultQuadrotor": {
                            "PawnBP": pawn_bp
                        }
                    }
                }
            else:
                # Read current settings
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                # Update version
                settings["SettingsVersion"] = 1.2
                # Create PawnPaths if missing
                if 'PawnPaths' not in settings:
                    settings['PawnPaths'] = {}
                if 'DefaultQuadrotor' not in settings['PawnPaths']:
                    settings['PawnPaths']['DefaultQuadrotor'] = {}
                # Update PawnBP
                settings['PawnPaths']['DefaultQuadrotor']['PawnBP'] = pawn_bp
            # Create backup
            backup_path = os.path.join(os.path.dirname(settings_path), "settings_backup.json")
            with open(backup_path, 'w') as f:
                json.dump(settings, f, indent=2)
            # Write updated settings
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"Updated AirSim settings to use drone type: {drone_type}")
            print(f"PawnBP set to: {pawn_bp}")
        except Exception as e:
            print(f"Error updating AirSim settings: {e}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parallel data collection')
    parser.add_argument('--config', type=str, default="scripts\Data_collection\data_collection_config\config8-brushify-lake.json", 
                        help='Path to configuration file')
    parser.add_argument('--setres', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                        help='Set resolution width and height (e.g., --setres 1080 720)')
    parser.add_argument('--state', action='store_true',
                        help='Enable state saving - save drone and camera ground truth data to text files')
    
    args = parser.parse_args()
    
    # Define the list of configuration files to process
    config_files_to_process = [
         
        "C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\scripts\Data_collection\main\config\bo_config-city.json",
        "C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\scripts\Data_collection\main\config\bo_config-forest-test4.json",
        "C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\scripts\Data_collection\main\config\bo_config-lake.json",
        "C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\scripts\Data_collection\main\config\bo_config-river.json",
        "C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\scripts\Data_collection\main\config\bo_config-rural.json",
    ]

    try:
        # Pass the first config file for initial setup of resolution and state saving
        # The ParallelDataCollection class will then load subsequent configs via _setup_run_parameters
        initial_config_for_setup = config_files_to_process[0]

        if args.setres:
            width, height = args.setres
            data_collection = ParallelDataCollection(initial_config_for_setup, width=width, height=height, save_state=args.state)
        else:
            data_collection = ParallelDataCollection(initial_config_for_setup, save_state=args.state)
        
        # Print state saving status
        if args.state:
            print("State saving ENABLED - Ground truth data will be saved to text files")
        else:
            print("State saving DISABLED - Only RGB and mask images will be saved")
        
        # Run the data collection process with all specified config files
        data_collection.run(config_files_to_process)
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal script execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")