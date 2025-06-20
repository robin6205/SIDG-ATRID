import json
import time
import signal
import sys
import os
import re
import traceback
import argparse
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
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Initialize drone type in AirSim settings
        drone_type = self.config['drone_config'].get('drone_type', 'DJIS900')
        self.init_drone_type(drone_type)
        
        # Initialize components
        self.unrealcv_client = Client(('127.0.0.1', 9000))
        
        # Connect to UnrealCV
        print("Connecting to UnrealCV...")
        self.unrealcv_client.connect()
        if not self.unrealcv_client.isconnected():
            raise ConnectionError("Could not connect to UnrealCV server")
        print("Successfully connected to UnrealCV")
        
        # Set resolution if provided
        if self.width is not None and self.height is not None:
            print(f"Setting resolution to {self.width}x{self.height}")
            self.unrealcv_client.request(f'r.setres {self.width}x{self.height}')
        
        # Try to connect to AirSim, but make it optional
        try:
            print("Connecting to AirSim...")
            self.airsim_client = MultirotorClient()
            max_retries = 10
            retry_delay = 3  # seconds
            
            for attempt in range(max_retries):
                try:
                    print(f"AirSim connection attempt {attempt+1}/{max_retries}")
                    self.airsim_client.confirmConnection()
                    print("Successfully connected to AirSim")
                    break
                except Exception as e:
                    print(f"Failed to connect to AirSim: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("AirSim connection failed. Will run in capture-only mode.")
                        self.airsim_client = None
        except Exception as e:
            print(f"Error setting up AirSim: {e}")
            self.airsim_client = None
        
        # Initialize data collection parameters
        base_dir = self.config['data_collection']['base_output_dir']
        location = self.config['data_collection']['location']
        self.base_output_dir = os.path.join(base_dir, drone_type, location)
        self.base_weather_condition = self.config['data_collection']['weather_condition']
        self.frame_rate = self.config['data_collection']['frame_rate']
        self.capture_duration = self.config['data_collection']['capture_duration']
        self.agent_list_path = "D:/Unreal Projects/ACREDataCollection/AgentList.json"
        self.shutting_down = False  # Add shutdown flag
        self.continue_capture = True  # Flag to control capture loop
        
        # Create output directories for each camera with their specific weather condition
        self.camera_dirs = {}
        for camera_id in self.config['camera_config'].keys():
            # Create weather condition string specific to this camera
            # e.g., "city-sunny-cam1" for camera1
            camera_num = camera_id.replace('camera', '')
            camera_weather_condition = self.base_weather_condition.replace('cam3', f'cam{camera_num}')
            
            # Create camera-specific weather output directory
            camera_weather_dir = os.path.join(self.base_output_dir, camera_weather_condition)
            
            # Set up directories for this camera
            self.camera_dirs[camera_id] = {
                'weather_dir': camera_weather_dir,
                'rgb': os.path.join(camera_weather_dir, 'rgb'),
                'mask': os.path.join(camera_weather_dir, 'mask')
            }
            
            # Add state directory if state saving is enabled
            if self.save_state:
                self.camera_dirs[camera_id]['state'] = os.path.join(camera_weather_dir, 'state')
                os.makedirs(self.camera_dirs[camera_id]['state'], exist_ok=True)
                print(f"Created state directory for {camera_id}: {self.camera_dirs[camera_id]['state']}")
            
            # Create directories
            os.makedirs(self.camera_dirs[camera_id]['rgb'], exist_ok=True)
            os.makedirs(self.camera_dirs[camera_id]['mask'], exist_ok=True)
            
            # Create agent color info file specific to this camera
            self.camera_dirs[camera_id]['agent_color_info'] = os.path.join(
                camera_weather_dir, "agent_color_info.json"
            )

        # Get the starting frame index
        self.frame_index = self._get_max_frame_index()
        print(f"Starting data collection from frame index: {self.frame_index}")
        
        # Register signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

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

    def set_agent_color(self, agent_list, target_agent_name, target_color):
        """Set a specific color for a target agent and handle one conflicting color"""
        # Pause simulation
        self.unrealcv_client.request('vset /action/game/pause')
        
        objects_response = self.unrealcv_client.request('vget /objects')
        all_objects = objects_response.split(' ')
        print(f"Found {len(all_objects)} objects in the scene")
        
        # Target color (yellow) to avoid
        yellow_color = (255, 255, 0)
        
        # Check each object's color and change if it matches yellow
        for obj_name in all_objects:
            if not obj_name or obj_name == 'Drone1':  # Skip empty strings and the drone
                continue
            
            try:
                # Get current color of the object
                color_str = self.unrealcv_client.request(f'vget /object/{obj_name}/color')
                match = re.match(r'\(R=(\d+),G=(\d+),B=(\d+),A=(\d+)\)', color_str)
                
                if match:
                    current_color = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                    
                    # If the object is yellow, change its color to a different color (e.g., blue)
                    if current_color == yellow_color:
                        print(f"Found conflicting yellow color on object: {obj_name}")
                        # Set to blue (0, 0, 255)
                        self.unrealcv_client.request(f'vset /object/{obj_name}/color 0 0 255')
                        print(f"Changed color of {obj_name} to blue")
                        break  # Exit after handling the first conflict
            except Exception as e:
                print(f"Error processing object {obj_name}: {e}")
                continue
        
        # Finally, set the drone to yellow
        self.unrealcv_client.request('vset /object/Drone1/color 255 255 0')
        print("Set Drone1 color to yellow")
        
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
            
            # Spawn and configure the camera
            print(f"Spawning camera {camera_num}...")
            spawn_result1 = self.unrealcv_client.request('vset /cameras/spawn')
            spawn_result2 = self.unrealcv_client.request('vset /cameras/spawn')
            print(f"Spawn results for camera {camera_num}: {spawn_result1}, {spawn_result2}")
            
            # Set camera location
            location_cmd = f'vset /camera/{camera_num}/location {camera_config["location"]["x"]} {camera_config["location"]["y"]} {camera_config["location"]["z"]}'
            location_result = self.unrealcv_client.request(location_cmd)
            print(f"Location command for camera {camera_num}: {location_cmd}")
            print(f"Location result: {location_result}")
            
            # Set camera rotation
            rotation_cmd = f'vset /camera/{camera_num}/rotation {camera_config["rotation"]["pitch"]} {camera_config["rotation"]["yaw"]} {camera_config["rotation"]["roll"]}'
            rotation_result = self.unrealcv_client.request(rotation_cmd)
            print(f"Rotation command for camera {camera_num}: {rotation_cmd}")
            print(f"Rotation result: {rotation_result}")
            
            print(f"{camera_id} set to location {camera_config['location']} and rotation {camera_config['rotation']}")
        
        # Debug: Check available cameras after setup
        camera_list_after = self.unrealcv_client.request('vget /cameras')
        print(f"Available cameras after setup: {camera_list_after}")
        print("All cameras set up successfully")

    def _initialize_drone(self):
        """Initialize drone and store initial position"""
        if self.airsim_client is None:
            print("AirSim not available. Skipping drone initialization.")
            return False
            
        self.airsim_client.enableApiControl(True)
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
        
        print(f"\nMoving to waypoint:")
        print(f"X: {x}m")
        print(f"Y: {y}m")
        print(f"Relative altitude: {relative_altitude}m")
        print(f"Velocity: {velocity}m/s")
        
        self.airsim_client.moveToPositionAsync(
            x=x,
            y=y,
            z=z,
            velocity=velocity
        ).join()

    def _move_to_geographic_waypoint(self, latitude, longitude, relative_altitude, velocity):
        """Move drone to specified waypoint using geographic coordinates"""
        if self.airsim_client is None:
            print("AirSim not available. Skipping geographic waypoint movement.")
            return
            
        target_altitude = self.initial_gps_altitude + relative_altitude
        
        print(f"\nMoving to waypoint:")
        print(f"Latitude: {latitude}")
        print(f"Longitude: {longitude}")
        print(f"Target altitude: {target_altitude:.2f}m (Initial + {relative_altitude}m)")
        print(f"Velocity: {velocity}m/s")
        
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
        
        print("Initiating return to home sequence...")
        
        try:
            # Ensure simulation is running
            if self.unrealcv_client.request('vget /action/game/is_paused') == 'true':
                self.unrealcv_client.request('vset /action/game/resume')
            
            # First move to a position 3m above the initial position
            print(f"Moving to position 3m above initial position...")
            self.airsim_client.moveToPositionAsync(
                x=self.initial_position.x_val,
                y=self.initial_position.y_val,
                z=self.initial_z - 3,  # 3m above initial height (remember z is negative in NED)
                velocity=5
            ).join()
            
            print("Returning to initial position...")
            # Then move to the exact initial position
            self.airsim_client.moveToPositionAsync(
                x=self.initial_position.x_val,
                y=self.initial_position.y_val,
                z=self.initial_z,
                velocity=2
            ).join()
            
            # Land from there
            print("Landing...")
            self.airsim_client.landAsync().join()
            
        except Exception as e:
            print(f"Error during return to home: {e}")
        
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
        
        # Get drone data from AirSim
        if self.airsim_client is not None:
            try:
                # Get drone position and orientation from AirSim
                drone_state = self.airsim_client.getMultirotorState()
                drone_position = drone_state.kinematics_estimated.position
                drone_orientation = drone_state.kinematics_estimated.orientation
                
                # Convert quaternion to euler angles (roll, pitch, yaw)
                import math
                # Quaternion to Euler conversion
                w, x, y, z = drone_orientation.w_val, drone_orientation.x_val, drone_orientation.y_val, drone_orientation.z_val
                
                # Roll (x-axis rotation)
                sinr_cosp = 2 * (w * x + y * z)
                cosr_cosp = 1 - 2 * (x * x + y * y)
                roll = math.atan2(sinr_cosp, cosr_cosp)
                
                # Pitch (y-axis rotation)
                sinp = 2 * (w * y - z * x)
                if abs(sinp) >= 1:
                    pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
                else:
                    pitch = math.asin(sinp)
                
                # Yaw (z-axis rotation)
                siny_cosp = 2 * (w * z + x * y)
                cosy_cosp = 1 - 2 * (y * y + z * z)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                
                # Convert radians to degrees
                roll_deg = math.degrees(roll)
                pitch_deg = math.degrees(pitch)
                yaw_deg = math.degrees(yaw)
                
                state_data['drone'] = {
                    'position': {
                        'x': drone_position.x_val,
                        'y': drone_position.y_val,
                        'z': drone_position.z_val
                    },
                    'rotation': {
                        'roll': roll_deg,
                        'pitch': pitch_deg,
                        'yaw': yaw_deg
                    }
                }
                
            except Exception as e:
                print(f"Error getting drone state from AirSim: {e}")
                state_data['drone'] = {
                    'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'rotation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                    'error': str(e)
                }
        else:
            print("AirSim not available for drone state")
            state_data['drone'] = {
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'rotation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                'error': 'AirSim not available'
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
        Save ground truth state data to a text file.
        Creates a file with the same naming pattern as the images.
        """
        if not self.save_state:
            return
            
        try:
            # Get ground truth data
            state_data = self._get_ground_truth_data(camera_id, timestamp, frame_index)
            
            # Create filename matching the image pattern
            state_filename = f"{frame_index}_{timestamp}_{camera_id}_state.txt"
            state_path = os.path.join(self.camera_dirs[camera_id]['state'], state_filename)
            
            # Write state data to text file
            with open(state_path, 'w') as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Frame Index: {frame_index}\n")
                f.write(f"Camera ID: {camera_id}\n")
                f.write("\n--- DRONE STATE ---\n")
                f.write(f"Position (m):\n")
                f.write(f"  X: {state_data['drone']['position']['x']:.6f}\n")
                f.write(f"  Y: {state_data['drone']['position']['y']:.6f}\n")
                f.write(f"  Z: {state_data['drone']['position']['z']:.6f}\n")
                f.write(f"Rotation (degrees):\n")
                f.write(f"  Roll:  {state_data['drone']['rotation']['roll']:.6f}\n")
                f.write(f"  Pitch: {state_data['drone']['rotation']['pitch']:.6f}\n")
                f.write(f"  Yaw:   {state_data['drone']['rotation']['yaw']:.6f}\n")
                
                f.write(f"\n--- CAMERA STATE ---\n")
                f.write(f"Camera Number: {state_data['camera'].get('camera_number', 'N/A')}\n")
                f.write(f"Position (Unreal units):\n")
                f.write(f"  X: {state_data['camera']['position']['x']:.6f}\n")
                f.write(f"  Y: {state_data['camera']['position']['y']:.6f}\n")
                f.write(f"  Z: {state_data['camera']['position']['z']:.6f}\n")
                f.write(f"Rotation (degrees):\n")
                f.write(f"  Roll:  {state_data['camera']['rotation']['roll']:.6f}\n")
                f.write(f"  Pitch: {state_data['camera']['rotation']['pitch']:.6f}\n")
                f.write(f"  Yaw:   {state_data['camera']['rotation']['yaw']:.6f}\n")
                
                # Add any error information
                if 'error' in state_data['drone']:
                    f.write(f"\nDrone Error: {state_data['drone']['error']}\n")
                if 'error' in state_data['camera']:
                    f.write(f"Camera Error: {state_data['camera']['error']}\n")
            
            print(f"Saved state data to: {state_path}")
            
        except Exception as e:
            print(f"Error saving state data for {camera_id}: {e}")

    def capture_frames_parallel(self):
        """Capture RGB and object mask frames from all cameras in parallel."""
        start_time = time.time()
        max_images = self.config['data_collection'].get('max_images', float('inf'))
        
        print(f"Starting parallel capture for all cameras, max images: {max_images}")
        
        # Reset frame counter
        self.frame_index = 0
        
        # Main capture loop
        while self.continue_capture and self.frame_index < max_images and (time.time() - start_time) < self.capture_duration:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Pause simulation
            self.unrealcv_client.request('vrun pause')
            
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
            self.unrealcv_client.request('vrun pause')
        
        # Determine why we stopped
        if self.frame_index >= max_images:
            print(f"Reached maximum frame count of {max_images}")
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
                print(f"  State: {dirs['state']} ({self.frame_index} text files)")

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

    def _setup_camera(self):
        """Set up all cameras and configure drone color"""
        # Set agent colors
        agent_list = self.load_agent_list()
        self.set_agent_color(agent_list, "Drone1", (255, 255, 0))
        
        # Get agent colors
        self.get_agent_colors(agent_list)
        
        # Set up all cameras
        self.setup_all_cameras()

    def _stop_capture(self):
        """Stop frame capture and ensure simulation is running"""
        self.continue_capture = False
        # Ensure simulation is running
        self.unrealcv_client.request('vset /action/game/resume')

    def run(self):
        """Run the parallel data collection process"""
        try:
            # Set up cameras and agent colors
            self._setup_camera()
            
            # Check if AirSim is available
            airsim_available = self.airsim_client is not None
            
            if airsim_available:
                print("Running in full drone + parallel capture mode")
                
                # Initialize and take off
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
                
                # Land and reset drone
                self._land_and_reset()
                
            else:
                print("AirSim not available. Running in capture-only mode.")
                
                # Just run the capture function directly
                self.capture_frames_parallel()
            
        except Exception as e:
            print(f"Error during mission: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            self._stop_capture()
        finally:
            try:
                # Land drone and cleanup if not already done
                if airsim_available:
                    self._land_and_reset()
            except Exception as e:
                print(f"Error during final landing: {e}")
            
            if self.unrealcv_client.isconnected():
                self.unrealcv_client.disconnect()
            print("Mission completed")

    def init_drone_type(self, drone_type):
        """
        Update the AirSim settings.json file to use the specified drone type.
        
        Args:
            drone_type (str): The type of drone to use (e.g., 'DJIS900', 'DJI_Phantom')
        """
        print(f"Initializing drone type: {drone_type}")
        
        # Path to AirSim settings.json
        settings_path = os.path.expanduser("~/Documents/AirSim/settings.json")
        
        try:
            # Check if settings file exists
            if not os.path.exists(settings_path):
                print(f"Warning: AirSim settings file not found at {settings_path}")
                return
            
            # Read current settings
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            
            # Create a backup of the original settings
            backup_path = os.path.join(os.path.dirname(settings_path), "settings_backup.json")
            with open(backup_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            # Update the PawnBP path based on drone_type
            if 'PawnPaths' in settings and 'DefaultQuadrotor' in settings['PawnPaths']:
                # Format the PawnBP string based on drone_type
                pawn_bp = f"Class '/AirSim/Blueprints/BP_{drone_type}.BP_{drone_type}_C'"
                settings['PawnPaths']['DefaultQuadrotor']['PawnBP'] = pawn_bp
                
                # Write updated settings back to file
                with open(settings_path, 'w') as f:
                    json.dump(settings, f, indent=2)
                
                print(f"Updated AirSim settings to use drone type: {drone_type}")
                print(f"PawnBP set to: {pawn_bp}")
            else:
                print("Warning: Could not find PawnPaths.DefaultQuadrotor in AirSim settings")
        
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
    
    try:
        if args.setres:
            width, height = args.setres
            data_collection = ParallelDataCollection(args.config, width=width, height=height, save_state=args.state)
        else:
            data_collection = ParallelDataCollection(args.config, save_state=args.state)
        
        # Print state saving status
        if args.state:
            print("State saving ENABLED - Ground truth data will be saved to text files")
        else:
            print("State saving DISABLED - Only RGB and mask images will be saved")
        
        data_collection.run()
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal script execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")