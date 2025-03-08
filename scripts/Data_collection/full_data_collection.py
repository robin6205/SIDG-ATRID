import json
import time
import signal
import sys
import os
import re
from datetime import datetime
from unrealcv import Client
from airsim import MultirotorClient
import threading

class FullDataCollection:
    def __init__(self, config_file):
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Initialize drone type in AirSim settings
        drone_type = self.config['drone_config'].get('drone_type', 'DJIS900')
        self.init_drone_type(drone_type)
        
        # Initialize components
        self.unrealcv_client = Client(('127.0.0.1', 9000))
        self.airsim_client = MultirotorClient()
        
        # Connect to clients
        self._connect_clients()
        
        # Initialize data collection parameters
        base_dir = self.config['data_collection']['base_output_dir']
        location = self.config['data_collection']['location']
        self.base_output_dir = os.path.join(base_dir, drone_type, location)
        self.base_weather_condition = self.config['data_collection']['weather_condition']
        self.frame_rate = self.config['data_collection']['frame_rate']
        self.capture_duration = self.config['data_collection']['capture_duration']
        self.agent_list_path = "D:/Unreal Projects/ACREDataCollection/AgentList.json"
        self.pause_count = 0  # Track number of pause commands
        self.shutting_down = False  # Add shutdown flag

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
            
            # Create directories
            os.makedirs(self.camera_dirs[camera_id]['rgb'], exist_ok=True)
            os.makedirs(self.camera_dirs[camera_id]['mask'], exist_ok=True)
            
            # Create agent color info file specific to this camera
            self.camera_dirs[camera_id]['agent_color_info'] = os.path.join(
                camera_weather_dir, "agent_color_info.json"
            )

        # Pause simulation and get the starting frame index
        # self.unrealcv_client.request('vrun pause')
        # use vset /action/game/pause
        self.unrealcv_client.request('vset /action/game/pause')
        self.frame_index = self._get_max_frame_index()
        print(f"Starting data collection from frame index: {self.frame_index}")
        
        # Resume simulation
        # self.unrealcv_client.request('vrun pause')
        # use vset /action/game/resume
        self.unrealcv_client.request('vset /action/game/resume')

        # Register signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle interrupt signal for immediate shutdown"""
        print('\nImmediate shutdown requested...')
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
        # self.unrealcv_client.request('vrun pause')
        # use vset /action/game/pause
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
                    
                    # print index number of the object
                    print(f"Object {obj_name} has color: {current_color}")
                    
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
        
        # self.unrealcv_client.request('vrun resume')
        # use vset /action/game/resume
        self.unrealcv_client.request('vset /action/game/resume')

    def set_camera_position(self):
        """Set up all cameras' positions and rotations."""
        for camera_id, camera_config in self.config['camera_config'].items():
            # Convert camera_id to numeric ID (e.g., 'camera1' -> 1)
            camera_num = int(camera_id.replace('camera', ''))
            
            # Spawn and configure each camera
            self.unrealcv_client.request('vset /cameras/spawn')
            self.unrealcv_client.request('vset /cameras/spawn')
            self.unrealcv_client.request(
                f'vset /camera/{camera_num}/location {camera_config["location"]["x"]} {camera_config["location"]["y"]} {camera_config["location"]["z"]}'
            )
            self.unrealcv_client.request(
                f'vset /camera/{camera_num}/rotation {camera_config["rotation"]["pitch"]} {camera_config["rotation"]["yaw"]} {camera_config["rotation"]["roll"]}'
            )
            print(f"{camera_id} set to location {camera_config['location']} and rotation {camera_config['rotation']}")

    def _pause_simulation(self):
        """Pause the simulation if running"""
        if self.pause_count % 2 == 0:
            # self.unrealcv_client.request('vrun pause')
            # use vset /action/game/pause
            self.unrealcv_client.request('vset /action/game/pause')
            self.pause_count += 1
            # Add a small delay to ensure the pause takes effect
            time.sleep(0.5)

    def _resume_simulation(self):
        """Resume the simulation if paused"""
        if self.pause_count % 2 != 0:
            # self.unrealcv_client.request('vrun pause')
            # use vset /action/game/resume
            self.unrealcv_client.request('vset /action/game/resume')
            self.pause_count += 1

    def capture_frames(self):
        """Capture RGB and object mask frames for all cameras."""
        start_time = time.time()
        max_images = self.config['data_collection'].get('max_images', float('inf'))
        image_count_consistency = self.config['data_collection'].get('image_count_consistency', False)
        
        # Track the number of images captured by the first camera
        first_camera_image_count = None
        
        while self.frame_index < max_images and (time.time() - start_time) < self.capture_duration:
            frame_start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Pause simulation and ensure it's fully paused
            self._pause_simulation()
            
            # wait for scene to render completely
            time.sleep(1)
            
            # Keep track of all files being written
            current_frame_files = []
            
            # Capture images from all cameras
            for camera_id in self.config['camera_config'].keys():
                camera_num = int(camera_id.replace('camera', ''))
                
                # If image count consistency is enabled and we have a first camera count,
                # skip if we've already captured enough images for this camera
                if image_count_consistency and first_camera_image_count is not None and self.frame_index >= first_camera_image_count:
                    continue
                    
                # Capture and save object mask image
                mask_filename = f"{self.frame_index}_{timestamp}_{camera_id}_object_mask.png"
                mask_path = os.path.join(self.camera_dirs[camera_id]['mask'], mask_filename)
                self.unrealcv_client.request(f'vget /camera/{camera_num}/object_mask {mask_path}')
                current_frame_files.append(mask_path)

                # Capture and save RGB image
                rgb_filename = f"{self.frame_index}_{timestamp}_{camera_id}_lit.png"
                rgb_path = os.path.join(self.camera_dirs[camera_id]['rgb'], rgb_filename)
                self.unrealcv_client.request(f'vget /camera/{camera_num}/lit {rgb_path}')
                current_frame_files.append(rgb_path)
                
                # print debug information
                print(f"Captured {self.frame_index} frames")
                print(f"Total images captured: {self.frame_index * len(self.camera_dirs) * 2}")  # *2 for RGB and mask
                for camera_id, dirs in self.camera_dirs.items():
                    print(f"{camera_id} images saved to:")
                    print(f"  RGB: {dirs['rgb']} ({self.frame_index} images)")
                    print(f"  Mask: {dirs['mask']} ({self.frame_index} images)")
            
            # Wait for all files to be written completely
            # wait_start = time.time()
            # max_wait_time = 15  # Increased from 10 to 15 seconds
            # all_files_saved = False
            
            # print(f"Waiting for {len(current_frame_files)} files to be saved...")
            
            # while not all_files_saved and (time.time() - wait_start) < max_wait_time:
            #     all_files_saved = True
            #     for file_path in current_frame_files:
            #         if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            #             all_files_saved = False
            #             break
            #     if not all_files_saved:
            #         time.sleep(0.1)
            
            # if not all_files_saved:
            #     print(f"Warning: Not all files were saved within {max_wait_time} seconds")
            #     # List which files weren't saved
            #     for file_path in current_frame_files:
            #         if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            #             print(f"  Missing or empty file: {file_path}")
            # else:
            #     print(f"All files for frame {self.frame_index} saved successfully")
                # If this is the first camera and we're at the max images, store the count
                if image_count_consistency and camera_id == 'camera1' and (self.frame_index + 1 >= max_images or (time.time() - start_time) >= self.capture_duration):
                    first_camera_image_count = self.frame_index + 1
                    print(f"First camera captured {first_camera_image_count} images. Other cameras will match this count.")
            
            self.frame_index += 1
                
            # Resume simulation
            self._resume_simulation()

            
            # Calculate how long this frame took
            frame_time = time.time() - frame_start_time
            target_frame_time = 1.0 / self.frame_rate
            
            # Sleep to maintain frame rate, if needed
            # sleep_time = max(0, target_frame_time - frame_time)
            # if sleep_time > 0:
            #     time.sleep(sleep_time)
            # else:
            #     print(f"Warning: Frame {self.frame_index-1} took {frame_time:.2f}s, exceeding target frame time of {target_frame_time:.2f}s")
        
        # Determine why we stopped
        if self.frame_index >= max_images:
            print(f"Reached maximum frame count of {max_images}")
        else:
            print(f"Reached capture duration limit of {self.capture_duration} seconds")
        
        print(f"Captured {self.frame_index} frames")
        print(f"Total images captured: {self.frame_index * len(self.camera_dirs) * 2}")  # *2 for RGB and mask
        for camera_id, dirs in self.camera_dirs.items():
            print(f"{camera_id} images saved to:")
            print(f"  RGB: {dirs['rgb']} ({self.frame_index} images)")
            print(f"  Mask: {dirs['mask']} ({self.frame_index} images)")

    def _connect_clients(self):
        """Connect to UnrealCV and AirSim clients"""
        self.unrealcv_client.connect()
        if not self.unrealcv_client.isconnected():
            raise ConnectionError("Could not connect to UnrealCV server")
        
        self.airsim_client.confirmConnection()

    def _initialize_drone(self):
        """Initialize drone and store initial position"""
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

    def _takeoff(self):
        """Perform takeoff sequence"""
        print("Taking off...")
        self.airsim_client.takeoffAsync().join()
        time.sleep(2)  # Stabilization delay

    def _move_to_waypoint(self, x, y, relative_altitude, velocity):
        """Move drone to specified waypoint with relative altitude"""
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

    def _land_and_reset(self):
        """Perform landing sequence and reset drone"""
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
        except Exception as e:
            print(f"Error resetting drone: {e}")

    def _print_current_position(self):
        """Print current position information"""
        state = self.airsim_client.getMultirotorState()
        position = state.kinematics_estimated.position
        
        print(f"Current position:")
        print(f"X: {position.x_val:.2f}m")
        print(f"Y: {position.y_val:.2f}m")
        print(f"Z: {position.z_val:.2f}m")
        print(f"Height above start: {-(position.z_val - self.initial_z):.2f}m")

    def _setup_camera(self):
        """Set up all cameras and configure drone color"""
        # Set agent colors
        agent_list = self.load_agent_list()
        self.set_agent_color(agent_list, "Drone1", (255, 255, 0))
        
        # # get agent colors
        # self.get_agent_colors(agent_list)

    def _setup_specific_camera(self, camera_id):
        """Set up a specific camera position and rotation"""
        print(f"Setting up {camera_id}...")
        
        # Get camera configuration
        camera_config = self.config['camera_config'][camera_id]
        
        # Convert camera_id to numeric ID (e.g., 'camera1' -> 1)
        camera_num = int(camera_id.replace('camera', ''))
        
        # Ensure simulation is running for camera setup
        if self.unrealcv_client.request('vget /action/game/is_paused') == 'true':
            self.unrealcv_client.request('vset /action/game/resume')
            print("Resumed simulation for camera setup")
        
        # Spawn and configure the camera
        print(f"Spawning camera {camera_num}...")
        spawn_result1 = self.unrealcv_client.request('vset /cameras/spawn')
        spawn_result2 = self.unrealcv_client.request('vset /cameras/spawn')
        print(f"Spawn results: {spawn_result1}, {spawn_result2}")
        
        # Set camera location
        location_cmd = f'vset /camera/{camera_num}/location {camera_config["location"]["x"]} {camera_config["location"]["y"]} {camera_config["location"]["z"]}'
        location_result = self.unrealcv_client.request(location_cmd)
        print(f"Location command: {location_cmd}")
        print(f"Location result: {location_result}")
        
        # Set camera rotation
        rotation_cmd = f'vset /camera/{camera_num}/rotation {camera_config["rotation"]["pitch"]} {camera_config["rotation"]["yaw"]} {camera_config["rotation"]["roll"]}'
        rotation_result = self.unrealcv_client.request(rotation_cmd)
        print(f"Rotation command: {rotation_cmd}")
        print(f"Rotation result: {rotation_result}")
        
        # Verify camera exists
        camera_list = self.unrealcv_client.request('vget /cameras')
        print(f"Available cameras after setup: {camera_list}")
        
        print(f"{camera_id} set to location {camera_config['location']} and rotation {camera_config['rotation']}")
        
        # Reset pause count to ensure consistent state
        self.pause_count = 0

    def _execute_mission(self):
        """Execute the drone mission based on the coordinate system"""
        mission_file = self.config['drone_config']['mission_file']
        
        with open(mission_file, 'r') as f:
            mission_data = json.load(f)
        
        waypoints = mission_data['drones'][0]['waypoints']
        
        # Initialize and take off
        self._initialize_drone()
        self._takeoff()
        
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
        
        # Land and cleanup
        # self._land_and_reset()
        # print("Mission completed successfully")

    def _move_to_geographic_waypoint(self, latitude, longitude, relative_altitude, velocity):
        """Move drone to specified waypoint using geographic coordinates"""
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

    def _print_current_altitude(self):
        """Print current altitude information"""
        state = self.airsim_client.getMultirotorState()
        ned_altitude = -state.kinematics_estimated.position.z_val
        gps = self.airsim_client.getGpsData()
        gps_altitude = gps.gnss.geo_point.altitude
        
        print(f"Current NED altitude: {ned_altitude:.2f}m")
        print(f"Current GPS altitude: {gps_altitude:.2f}m")
        print(f"Height above start: {gps_altitude - self.initial_gps_altitude:.2f}m")

    def _start_capture_for_camera(self, target_camera_id):
        """Start frame capture for a specific camera in a separate thread"""
        self.target_camera_id = target_camera_id
        self.capture_thread = threading.Thread(target=self.capture_frames_for_camera)
        self.capture_thread.start()

    def capture_frames_for_camera(self):
        """Capture RGB and object mask frames for the target camera only."""
        start_time = time.time()
        max_images = self.config['data_collection'].get('max_images', float('inf'))
        
        print(f"Starting capture for {self.target_camera_id}, max images: {max_images}")
        
        # Flag to track if we should continue capturing
        self.continue_capture = True
        
        # Reset frame counter for this camera
        self.frame_index = 0
        
        while self.continue_capture and self.frame_index < max_images and (time.time() - start_time) < self.capture_duration:
            frame_start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Pause simulation and ensure it's fully paused
            self._pause_simulation()
            
            # Wait for scene to render completely
            time.sleep(1)
            
            # Keep track of all files being written
            current_frame_files = []
            
            # Get camera number
            camera_num = int(self.target_camera_id.replace('camera', ''))
            
            # Debug: Check if camera exists
            if self.frame_index == 0:
                camera_list = self.unrealcv_client.request('vget /cameras')
                print(f"Available cameras before capture: {camera_list}")
            
            # Capture and save object mask image
            mask_filename = f"{self.frame_index}_{timestamp}_{self.target_camera_id}_object_mask.png"
            mask_path = os.path.join(self.camera_dirs[self.target_camera_id]['mask'], mask_filename)
            mask_cmd = f'vget /camera/{camera_num}/object_mask {mask_path}'
            mask_result = self.unrealcv_client.request(mask_cmd)
            
            if self.frame_index == 0:
                print(f"Mask command: {mask_cmd}")
                print(f"Mask result: {mask_result}")
            
            current_frame_files.append(mask_path)

            # Capture and save RGB image
            rgb_filename = f"{self.frame_index}_{timestamp}_{self.target_camera_id}_lit.png"
            rgb_path = os.path.join(self.camera_dirs[self.target_camera_id]['rgb'], rgb_filename)
            rgb_cmd = f'vget /camera/{camera_num}/lit {rgb_path}'
            rgb_result = self.unrealcv_client.request(rgb_cmd)
            
            if self.frame_index == 0:
                print(f"RGB command: {rgb_cmd}")
                print(f"RGB result: {rgb_result}")
            
            current_frame_files.append(rgb_path)
            
            # Check if files were created
            if self.frame_index % 10 == 0:
                mask_exists = os.path.exists(mask_path)
                rgb_exists = os.path.exists(rgb_path)
                print(f"Frame {self.frame_index} - Mask file exists: {mask_exists}, RGB file exists: {rgb_exists}")
            
            # Wait for all files to be written completely
            wait_start = time.time()
            max_wait_time = 15  # Increased from 10 to 15 seconds
            all_files_saved = False
            
            # print(f"Waiting for {len(current_frame_files)} files to be saved...")
            
            # while not all_files_saved and (time.time() - wait_start) < max_wait_time:
            #     all_files_saved = True
            #     for file_path in current_frame_files:
            #         if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            #             all_files_saved = False
            #             break
            #     if not all_files_saved:
            #         time.sleep(0.1)
            
            # if not all_files_saved:
            #     print(f"Warning: Not all files were saved within {max_wait_time} seconds")
            #     # List which files weren't saved
            #     for file_path in current_frame_files:
            #         if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            #             print(f"  Missing or empty file: {file_path}")
            # else:
            #     print(f"All files for frame {self.frame_index} saved successfully")
            
            self.frame_index += 1
            
            # Resume simulation
            self._resume_simulation()
            
            # Calculate how long this frame took
            frame_time = time.time() - frame_start_time
            target_frame_time = 1.0 / self.frame_rate
            
            # # Sleep to maintain frame rate, if needed
            # sleep_time = max(0, target_frame_time - frame_time)
            # if sleep_time > 0:
            #     time.sleep(sleep_time)
            # else:
            #     print(f"Warning: Frame {self.frame_index-1} took {frame_time:.2f}s, exceeding target frame time of {target_frame_time:.2f}s")
        
        # Determine why we stopped
        if self.frame_index >= max_images:
            print(f"Reached maximum frame count of {max_images}")
        elif not self.continue_capture:
            print("Data collection stopped because mission completed")
        else:
            print(f"Reached capture duration limit of {self.capture_duration} seconds")
        
        print(f"Captured {self.frame_index} frames for {self.target_camera_id}")
        print(f"Total images captured: {self.frame_index * 2}")  # *2 for RGB and mask
        print(f"{self.target_camera_id} images saved to:")
        print(f"  RGB: {self.camera_dirs[self.target_camera_id]['rgb']} ({self.frame_index} images)")
        print(f"  Mask: {self.camera_dirs[self.target_camera_id]['mask']} ({self.frame_index} images)")

    def _stop_capture(self):
        """Stop frame capture and ensure simulation is running"""
        self.continue_capture = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish
        # Ensure simulation is running with ispaused
        if self.unrealcv_client.request('vget /action/game/is_paused'):
            self.unrealcv_client.request('vset /action/game/resume')
        # if self.pause_count % 2 != 0:
        #     self.unrealcv_client.request('vrun pause')
        #     self.pause_count += 1

    def run(self):
        try:
            # Set up initial camera and agent colors
            self._setup_camera()
            
            # Process each camera in series
            for camera_id in self.config['camera_config'].keys():
                print(f"\n=== Starting data collection for {camera_id} ===")
                
                # Set up this specific camera
                self._setup_specific_camera(camera_id)
                
                # Reset frame index for each camera
                self.frame_index = self._get_max_frame_index()
                print(f"Starting data collection from frame index: {self.frame_index}")
                
                # Initialize and take off
                self._initialize_drone()
                self._takeoff()
                
                # Start data collection for this camera only
                self._start_capture_for_camera(camera_id)
                
                # Execute mission
                self._execute_mission()
                
                # Stop capture before landing
                self._stop_capture()
                
                # Land drone and reset for next camera
                self._land_and_reset()
                
                print(f"=== Completed data collection for {camera_id} ===\n")
                
                # Short pause between camera runs
                time.sleep(3)
            
        except Exception as e:
            print(f"Error during mission: {e}")
            self._stop_capture()
        finally:
            try:
                # Land drone and cleanup if not already done
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
    config_file = "scripts/Data_collection/data_collection_config/config4-park2.json"
    data_collection = FullDataCollection(config_file)
    data_collection.run() 