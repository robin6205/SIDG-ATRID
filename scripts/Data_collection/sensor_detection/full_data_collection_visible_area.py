import json
import time
import signal
import sys
import os
import re
import argparse
import subprocess
import shutil
from datetime import datetime
from unrealcv import Client
from airsim import MultirotorClient, Vector3r, DrivetrainType, YawMode, Quaternionr
import threading
import queue
import numpy as np
import math
import cv2
import glob  # Added for finding subset files
from pathlib import Path  # Added for path handling
"""
===============================================================================
File Name   : full_data_collection_visible_area.py
Description : Collects data from all cameras in a fixed configuration for a 
              specified duration. Saves RGB, mask, and state images for each camera.
              Also saves agent color information for each camera.
Author      : Josh Chang <chang529@purdue.edu>
Created On  : 2025-07-21
Last Updated: 2025-07-21
Version     : 1.2.3

Usage       : python full_data_collection_visible_area.py
Example     : python full_data_collection_visible_area.py --config_file "scripts/Data_collection/data_collection_config/generated_mission_data_debug_2.json" --save_state --visualize_line --parallel_mode

Dependencies:
    - Python >= 3.8
    - random, itertools, json, os, argparse, subprocess, shutil, datetime, unrealcv, airsim, threading, queue

Notes:
    - Generates 100 unique scenarios from all possible combinations
    - Each scenario includes camera configurations and environment conditions
    - Camera quality gradient: 1 (best) to 4 (worst)
    - Exposure levels range from -3 to 3
    - Focal lengths range from 10 to 2000
    - Outputs JSON files with scenario configurations
===============================================================================
"""
# Conversion scale: Unreal Editor (cm) -> AirSim (m)
SCALE = 0.01

# Hardcoded drone world position (meters) from drone_move_test.py
DRONE_WORLD_ORIGIN = Vector3r(
    -31173.356083 * SCALE,
    -72766.596592 * SCALE,
    331.905324 * SCALE
)

# Helper function for coordinate conversion
def world_to_ned(target: Vector3r, origin: Vector3r):
    return Vector3r(
        target.x_val - origin.x_val,
        target.y_val - origin.y_val,
        -(target.z_val - origin.z_val)
    )

class FullDataCollection:
    def __init__(self, config_file, save_state=False, visualize_line=False, parallel_mode=False, z_folder=None, launch_all_parallel=False):
        # Add z_folder parameter to store the z# folder path
        self.z_folder = z_folder
        self.subset_files = []
        self.current_subset_index = 0
        self.current_subset_config = None
        self.launch_all_parallel = launch_all_parallel
        self.all_subset_cameras = {}  # Will store all cameras from all subsets
        
        # Load subset files if z_folder is provided
        if self.z_folder:
            self._load_subset_files()
        
        # Add parallel_mode parameter to store the mode
        self.parallel_mode = parallel_mode
        print(f"Operating in {'PARALLEL' if parallel_mode else 'SEQUENTIAL'} mode")
        
        # Store state saving flag
        self.save_state = save_state
        self.visualize_line = visualize_line
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Extract cameras from config - prioritize camera_config for placement data
        if "camera_config" in self.config:
            # Use camera_config format which contains placement1-4 for subset mode
            self.camera_config = self.config["camera_config"]
            print(f"Using camera_config from config. Found {len(self.camera_config)} cameras.")
            print(f"Camera config keys: {list(self.camera_config.keys())}")
        elif "cameras" in self.config:
            # Fallback to cameras format if present
            if isinstance(self.config["cameras"], list):
                self.camera_config = {}
                for camera_obj in self.config["cameras"]:
                    # Each object in the list is a dict with a single camera
                    for camera_id, camera_data in camera_obj.items():
                        self.camera_config[camera_id] = camera_data
                
                print(f"Using cameras from list format. Found {len(self.camera_config)} cameras.")
            else:
                print("Warning: 'cameras' was not a list. Using empty camera config.")
                self.camera_config = {}
        else:
            # If neither format is present, initialize with empty dict
            self.camera_config = {}
            print("Warning: No camera configuration found in the config file.")
        
        # Check if the new JSON schema is used (with "mission" wrapper)
        if "mission" in self.config:
            # Extract cameras from mission section if not already extracted
            if "cameras" in self.config["mission"] and not self.camera_config:
                # Similar extraction logic as above
                if isinstance(self.config["mission"]["cameras"], list):
                    self.camera_config = {}
                    for camera_obj in self.config["mission"]["cameras"]:
                        for camera_id, camera_data in camera_obj.items():
                            self.camera_config[camera_id] = camera_data
                    print(f"Using cameras from mission wrapper. Found {len(self.camera_config)} cameras.")
            
            # Move mission data to top level for compatibility
            for key, value in self.config["mission"].items():
                if key != "cameras":  # Avoid duplicate with camera_config
                    self.config[key] = value
            
            # Remove the mission wrapper to maintain compatibility
            del self.config["mission"]
        
        # Print the cameras we found
        print(f"Camera configuration contains: {', '.join(self.camera_config.keys())}")
        
        # Initialize drone type in AirSim settings
        drone_type = self.config['drone_config'].get('drone_type', 'DJIS900')
        self.init_drone_type(drone_type)
        
        # Initialize components
        self.unrealcv_client = Client(('127.0.0.1', 9000))
        
        # Initialize AirSim client - always needed for drone control
        self.airsim_client = None  # Will initialize in connect_clients
        self.airsim_queue = queue.Queue()
        self.initial_position = None # Initialize to None
        self.initial_z = None        # Initialize to None
        self.initial_gps_altitude = None # Initialize to None
        
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
        self.shutting_down = False  # Add shutdown flag
        
        # Extract camera base pattern (removing "cam#" suffix if present)
        # Example: "urban-clearcheck-cam3" -> "urban-clearcheck-cam"
        weather_pattern = self.base_weather_condition
        if re.search(r'cam\d+$', weather_pattern):
            # If it ends with cam followed by digits, remove the digits but keep 'cam'
            weather_pattern = re.sub(r'cam\d+$', 'cam', weather_pattern)
        else:
            # If no cam pattern, append 'cam'
            weather_pattern = weather_pattern + "-cam"
        
        self.weather_base_pattern = weather_pattern  # Store for later use
        print(f"Base weather pattern: {self.weather_base_pattern}")
        
        # Find existing camera folders to avoid conflicts
        existing_camera_dirs = self._find_existing_camera_dirs()
        print(f"Existing camera directories: {existing_camera_dirs}")
        
        # Create output directories for each camera
        self.camera_dirs = {}
        
        # Only create traditional camera directories if NOT using subset mode
        if not self.z_folder:
            for camera_id in self.camera_config.keys():
                camera_num = int(camera_id.replace('camera', ''))
                
                # Find next available directory number
                dir_num = self._get_next_available_camera_number(camera_num, existing_camera_dirs)
                camera_weather_condition = f"{self.weather_base_pattern}{dir_num}"
                existing_camera_dirs.add(dir_num)  # Add to set so next cameras won't use this number
                
                camera_weather_dir = os.path.join(self.base_output_dir, camera_weather_condition)
                self.camera_dirs[camera_id] = {
                    'weather_dir': camera_weather_dir,
                    'rgb': os.path.join(camera_weather_dir, 'rgb'),
                    'mask': os.path.join(camera_weather_dir, 'mask'),
                    'state': os.path.join(camera_weather_dir, 'state') if self.save_state else None
                }
                
                # Create base directories
                os.makedirs(self.camera_dirs[camera_id]['rgb'], exist_ok=True)
                os.makedirs(self.camera_dirs[camera_id]['mask'], exist_ok=True)
                if self.save_state:
                    os.makedirs(self.camera_dirs[camera_id]['state'], exist_ok=True)
                
                self.camera_dirs[camera_id]['agent_color_info'] = os.path.join(
                    camera_weather_dir, "agent_color_info.json"
                )
                
                print(f"Created directory for {camera_id}: {camera_weather_dir}")
        else:
            print("Using subset mode - camera directories will be created per subset")
        
        # Load subset files if z_folder is provided
        if self.z_folder:
            self._load_subset_files()
        
        # Run postprocess_weatherconfig.py and copy the formatted_ultra_dynamic_sky.json to each camera directory
        self._run_postprocess_and_copy_weather_config()

        # Get starting frame index
        self.frame_index = self._get_max_frame_index()
        print(f"Starting data collection from frame index: {self.frame_index}")
        
        # Register signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

    def _load_subset_files(self):
        """Load subset files from the z_folder"""
        if self.z_folder:
            try:
                import glob
                import re
                subset_pattern = os.path.join(self.z_folder, 'subset_*.json')
                subset_files = glob.glob(subset_pattern)
                
                # Sort numerically instead of alphabetically
                def extract_number(filename):
                    match = re.search(r'subset_(\d+)\.json', os.path.basename(filename))
                    return int(match.group(1)) if match else 0
                
                self.subset_files = sorted(subset_files, key=extract_number)
                print(f"Loaded {len(self.subset_files)} subset files from {self.z_folder}")
                if self.subset_files:
                    print(f"First few files: {[os.path.basename(f) for f in self.subset_files[:5]]}")
            except Exception as e:
                print(f"Error loading subset files: {e}")
        else:
            print("No z_folder specified, skipping subset file loading")

    def _run_postprocess_and_copy_weather_config(self):
        """Run postprocess_weatherconfig.py and copy the formatted_ultra_dynamic_sky.json to each camera directory"""
        print("\n=== Processing and copying weather configuration ===")
        
        # Path to the postprocess_weatherconfig.py script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        postprocess_script_path = os.path.join(script_dir, "WeatherConfig", "postprocess_weatherconfig.py")
        
        # Target formatted JSON file path
        formatted_json_path = os.path.join(script_dir, "WeatherConfig", "formatted_ultra_dynamic_sky.json")
        
        # Check if files exist
        if not os.path.exists(postprocess_script_path):
            print(f"Warning: Weather config script not found at {postprocess_script_path}")
            return
            
        # Run the postprocess_weatherconfig.py script
        try:
            print(f"Running weather config processing script: {postprocess_script_path}")
            subprocess.run([sys.executable, postprocess_script_path], check=True)
            print("Weather config processing completed successfully")
            
            # Check if the formatted JSON file was created
            if not os.path.exists(formatted_json_path):
                print(f"Warning: Formatted weather config not found at {formatted_json_path}")
                return
                
            # Copy the formatted JSON to each camera directory
            for camera_id, dirs in self.camera_dirs.items():
                target_path = os.path.join(dirs['weather_dir'], "formatted_ultra_dynamic_sky.json")
                shutil.copy2(formatted_json_path, target_path)
                print(f"Copied weather config to {camera_id} directory: {target_path}")
                
                # Create camera_config.json file with resolution and FOV information
                self._create_camera_config_json(camera_id, dirs['weather_dir'])
                
            print("Weather configuration copied to all camera directories")
        except Exception as e:
            print(f"Error during weather config processing/copying: {e}")
            
        print("=== Weather configuration processing complete ===\n")
        
    def _create_camera_config_json(self, camera_id, target_dir):
        """
        Create a camera_config.json file with resolution, FOV, position and orientation information
        from the camera config. When using subset configurations, also includes exposure and focal length.
        
        Args:
            camera_id (str): The camera ID (e.g., 'camera1')
            target_dir (str): The directory to save the camera_config.json file
        """
        try:
            print(f"Creating camera_config.json for {camera_id}...")
            
            # Initialize with default values
            width = 1920
            height = 1080
            fov = 90
            position = {"x": 0, "y": 0, "z": 0}
            orientation = {"pitch": 0, "yaw": 0, "roll": 0}
            exposure = 0.0
            focal_length = 1000.0
            placement_number = None
            
            # Check if we're using subset configuration
            if self.current_subset_config:
                print(f"Using subset configuration for {camera_id}...")
                
                # Parse subset configuration to find this camera
                camera_configs = self._parse_camera_config_from_subset(self.current_subset_config)
                
                # Look for this camera in the parsed configs
                camera_found = False
                for cam_name, config in camera_configs.items():
                    if cam_name == camera_id or config["camera_id"] == int(camera_id.replace('camera', '')):
                        print(f"Found camera configuration in subset for {camera_id}")
                        
                        # Get values from subset configuration
                        width = config["resolution"]["width"]
                        height = config["resolution"]["height"]
                        fov = config["fov"]
                        exposure = config["exposure"]
                        focal_length = config["focal_length"]
                        placement_number = config["placement_number"]
                        
                        # Get position and orientation from placement
                        placement = config["placement"]
                        position = placement["location"]
                        orientation = placement["rotation"]
                        
                        print(f"Subset config - Resolution: {width}x{height}, FOV: {fov}, Exposure: {exposure}, Focal: {focal_length}")
                        print(f"Subset config - Position: {position}, Orientation: {orientation}")
                        
                        camera_found = True
                        break
                
                if not camera_found:
                    print(f"Warning: {camera_id} not found in subset configuration, falling back to original config")
            
            # Fallback to original camera config behavior if not using subset or camera not found
            if not self.current_subset_config or not camera_found:
                # Try to get values from camera config
                if camera_id in self.camera_config:
                    camera_data = self.camera_config[camera_id]
                    
                    # Check if specs/resolution exists
                    if "specs" in camera_data and "resolution" in camera_data["specs"]:
                        width = camera_data["specs"]["resolution"]["width"]
                        height = camera_data["specs"]["resolution"]["height"]
                        print(f"Using resolution from camera config: {width}x{height}")
                    # Also check direct resolution key
                    elif "resolution" in camera_data:
                        width = camera_data["resolution"]["width"]
                        height = camera_data["resolution"]["height"]
                        print(f"Using resolution from camera config: {width}x{height}")
                    
                    # Check if specs/fov exists
                    if "specs" in camera_data and "fov" in camera_data["specs"]:
                        fov = camera_data["specs"]["fov"]
                        print(f"Using FOV from camera config: {fov}")
                    # Also check direct fov key
                    elif "fov" in camera_data:
                        fov = camera_data["fov"]
                        print(f"Using FOV from camera config: {fov}")
                    
                    # Get position (check both 'position' and 'location' keys)
                    if "position" in camera_data:
                        position = camera_data["position"]
                        print(f"Using position from camera config: {position}")
                    elif "location" in camera_data:
                        position = camera_data["location"]
                        print(f"Using location from camera config: {position}")
                    
                    # Get orientation (check both 'orientation' and 'rotation' keys)
                    if "orientation" in camera_data:
                        orientation = camera_data["orientation"]
                        print(f"Using orientation from camera config: {orientation}")
                    elif "rotation" in camera_data:
                        orientation = camera_data["rotation"]
                        print(f"Using rotation from camera config: {orientation}")
                else:
                    print(f"Warning: {camera_id} not found in camera config, using default values: 1920x1080, FOV=90")
                    
                    # Only as a fallback, try to read from unrealcv.ini if it exists
                    unrealcv_ini_path = "D:/Program Files/Epic Games/UE_5.4/Engine/Binaries/Win64/unrealcv.ini"
                    if os.path.exists(unrealcv_ini_path):
                        print(f"Reading configuration from {unrealcv_ini_path}")
                        with open(unrealcv_ini_path, 'r') as ini_file:
                            ini_content = ini_file.read()
                            
                            # Parse width
                            width_match = re.search(r'Width=(\d+)', ini_content)
                            if width_match:
                                width = int(width_match.group(1))
                            
                            # Parse height
                            height_match = re.search(r'Height=(\d+)', ini_content)
                            if height_match:
                                height = int(height_match.group(1))
                            
                            # Parse FOV
                            fov_match = re.search(r'FOV=(\d+)', ini_content)
                            if fov_match:
                                fov = int(fov_match.group(1))
            
            # Create camera config JSON structure with all information
            camera_config_data = {
                "resolution": {
                    "width": width,
                    "height": height
                },
                "fov_deg": fov,
                "position": position,
                "orientation": orientation,
                "exposure": exposure,
                "focal_length": focal_length
            }
            
            # Add placement number if available (from subset config)
            if placement_number is not None:
                camera_config_data["placement_number"] = placement_number
            
            # Save to JSON file
            camera_config_path = os.path.join(target_dir, "camera_config.json")
            with open(camera_config_path, 'w') as f:
                json.dump(camera_config_data, f, indent=2)
                
            print(f"Created camera_config.json for {camera_id} at {camera_config_path}")
        except Exception as e:
            print(f"Error creating camera_config.json for {camera_id}: {e}")

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
            
            # Extract camera position (handle both old and new schemas)
            location = {}
            if "location" in camera_config:
                location = camera_config["location"]
            elif "position" in camera_config:
                location = camera_config["position"]
            
            # Extract camera rotation (handle both old and new schemas)
            rotation = {}
            if "rotation" in camera_config:
                rotation = camera_config["rotation"]
            elif "orientation" in camera_config:
                rotation = camera_config["orientation"]
            
            # Extract camera specs (handle both old and new schemas)
            resolution = None
            fov = None
            
            if "resolution" in camera_config:
                resolution = camera_config["resolution"]
            elif "specs" in camera_config and "resolution" in camera_config["specs"]:
                resolution = camera_config["specs"]["resolution"]
            
            if "fov" in camera_config:
                fov = camera_config["fov"]
            elif "specs" in camera_config and "fov" in camera_config["specs"]:
                fov = camera_config["specs"]["fov"]
            
            # Spawn and configure each camera
            self.unrealcv_client.request('vset /cameras/spawn')
            self.unrealcv_client.request('vset /cameras/spawn')
            self.unrealcv_client.request(
                f'vset /camera/{camera_num}/location {location["x"]} {location["y"]} {location["z"]}'
            )
            self.unrealcv_client.request(
                f'vset /camera/{camera_num}/rotation {rotation["pitch"]} {rotation["yaw"]} {rotation["roll"]}'
            )
            
            # Set camera resolution if specified in config
            if resolution:
                self.unrealcv_client.request(
                    f'vset /camera/{camera_num}/size {resolution["width"]} {resolution["height"]}'
                )
            
            # Set camera FOV if specified in config using the new method
            if fov:
                # Use the new set_cam_fov method
                self.set_cam_fov(camera_num, fov)
                
            print(f"{camera_id} set to location {location}, rotation {rotation}, "
                  f"resolution {resolution}, FOV {fov}")

    def set_cam_fov(self, cam_id, fov):
        """
        Set the camera field of view (fov).

        Parameters:
            cam_id (int): The camera ID.
            fov (float): The field of view in degrees.
        
        Example:
            set_cam_fov(0, 90)  # Set the fov of camera 0 to 90 degrees
        """
        try:
            print(f"Setting camera {cam_id} FOV to {fov} degrees")
            result = self.unrealcv_client.request(f'vset /camera/{cam_id}/fov {fov}')
            print(f"  Result: {result}")
            return result
        except Exception as e:
            print(f"Error setting FOV for camera {cam_id}: {e}")
            return None

    def _pause_simulation(self):
        """Pause the simulation if running"""
        # Simplified pause method - just for compatibility with other code
        self.unrealcv_client.request('vset /action/game/pause')
        # No need for pause counter or sleep

    def _resume_simulation(self):
        """Resume the simulation if paused"""
        # Simplified resume method - just for compatibility with other code
        self.unrealcv_client.request('vset /action/game/resume')
        # No need for pause counter

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
                print(f"Using camera number: {camera_num} for {camera_id}")
                
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
            #     print(f"All files for frame {self.frame_index-1} saved successfully")
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
        # Connect to UnrealCV
        print("Connecting to UnrealCV...")
        self.unrealcv_client.connect()
        if not self.unrealcv_client.isconnected():
            raise ConnectionError("Could not connect to UnrealCV server")
        print("Successfully connected to UnrealCV")
        
        # Connect to AirSim - always needed for drone control
        print("Connecting to AirSim...")
        max_retries = 10
        retry_delay = 3  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"AirSim connection attempt {attempt+1}/{max_retries}")
                # Initialize the main client instance here
                self.airsim_client = MultirotorClient() 
                self.airsim_client.confirmConnection()
                print("Successfully connected to AirSim")
                return
            except Exception as e:
                print(f"Failed to connect to AirSim: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.airsim_client = None
                    print("Failed to connect to AirSim after multiple attempts. Cannot control drone.")
        
    def _initialize_drone(self):
        """Initialize drone and store initial position"""
        print("\n=== INITIALIZING DRONE ===")
        
        # Check if we're able to control the drone
        try:
            control_success = self.airsim_client.enableApiControl(True)
            print(f"API control enabled: {control_success}")
        except Exception as e:
            print(f"Error enabling API control: {e}")
        
        # Try to arm the drone
        try:
            arm_success = self.airsim_client.armDisarm(True)
            print(f"Drone armed: {arm_success}")
        except Exception as e:
            print(f"Error arming drone: {e}")
        
        # Get and store initial position
        try:
            state = self.airsim_client.getMultirotorState()
            self.initial_position = state.kinematics_estimated.position
            self.initial_z = self.initial_position.z_val
            print(f"Initial position stored:")
            print(f"  X: {self.initial_position.x_val:.2f}m")
            print(f"  Y: {self.initial_position.y_val:.2f}m")
            print(f"  Z: {self.initial_z:.2f}m")
        except Exception as e:
            print(f"Error getting initial position: {e}")
            # Set default values in case of error
            print("Setting default initial position values")
            from airsim import Vector3r
            self.initial_position = Vector3r(0, 0, 0)
            self.initial_z = 0
        
        # Get initial GPS data
        try:
            gps_data = self.airsim_client.getGpsData()
            self.initial_gps_altitude = gps_data.gnss.geo_point.altitude
            print(f"Initial GPS data:")
            print(f"  Latitude: {gps_data.gnss.geo_point.latitude:.6f}")
            print(f"  Longitude: {gps_data.gnss.geo_point.longitude:.6f}")
            print(f"  Altitude: {self.initial_gps_altitude:.2f}m")
        except Exception as e:
            print(f"Error getting GPS data: {e}")
            self.initial_gps_altitude = 0
            
        print("=== DRONE INITIALIZATION COMPLETE ===\n")

    def _takeoff(self):
        """Perform takeoff sequence"""
        print("\n=== DRONE TAKEOFF SEQUENCE ===")
        try:
            print("Sending takeoff command...")
            takeoff_task = self.airsim_client.takeoffAsync()
            print("Waiting for takeoff completion...")
            takeoff_task.join()  # Removed timeout parameter
            print("Takeoff command completed")
            
            # Verify if takeoff was successful by checking altitude
            state = self.airsim_client.getMultirotorState()
            current_alt = -state.kinematics_estimated.position.z_val - (-self.initial_z)
            print(f"Current altitude after takeoff: {current_alt:.2f}m")
            
            print("Waiting 2 seconds for stabilization...")
            time.sleep(2)  # Stabilization delay
            print("Stabilization complete")
        except Exception as e:
            print(f"Error during takeoff: {e}")
            
        print("=== TAKEOFF SEQUENCE COMPLETE ===\n")

    def _move_to_waypoint(self, x, y, altitude, velocity):
        """Move drone to specified waypoint using NED coordinates relative to a fixed origin."""
        
        # Convert target waypoint (from mission file, assumed cm) to world coordinates (meters)
        target_world = Vector3r(
            x * SCALE,
            y * SCALE,
            altitude * SCALE
        )
        
        # Convert target world coordinates to NED relative to the hardcoded drone origin
        target_ned = world_to_ned(target_world, DRONE_WORLD_ORIGIN)
        
        print(f"\n=== DRONE MOVEMENT DETAILS ===")
        print(f"TARGET WAYPOINT (World, m): X={target_world.x_val:.2f}, Y={target_world.y_val:.2f}, Z={target_world.z_val:.2f}")
        print(f"REFERENCE ORIGIN (World, m): X={DRONE_WORLD_ORIGIN.x_val:.2f}, Y={DRONE_WORLD_ORIGIN.y_val:.2f}, Z={DRONE_WORLD_ORIGIN.z_val:.2f}")
        print(f"MOVEMENT VECTOR (NED, m): X={target_ned.x_val:.2f}, Y={target_ned.y_val:.2f}, Z={target_ned.z_val:.2f}")
        print(f"Velocity: {velocity}m/s")
        
        print(f"Sending moveToPositionAsync command...")
        try:
            # Execute the move command and wait for completion
            print(f"Executing drone movement...")
            move_task = self.airsim_client.moveToPositionAsync(
                x=target_ned.x_val,
                y=target_ned.y_val,
                z=target_ned.z_val -30,
                velocity=velocity
            )
            
            # move_task = self.airsim_client.moveToPositionAsync(
            #     x=target_ned.x_val,
            #     y=target_ned.y_val,
            #     z=target_ned.z_val -30,
            #     velocity=15,
            #     drivetrain=DrivetrainType.MaxDegreeOfFreedom,
            #     lookahead=-1,
            #     adaptive_lookahead=True
            # )

            
            print(f"Waiting for movement to complete...")
            move_task.join() 
            
            print(f"Movement command completed")
            
            # Check final position for debugging
            end_pose = self.airsim_client.simGetVehiclePose()
            print(f"FINAL POSITION - After movement (World, m):")
            print(f"  X: {end_pose.position.x_val:.2f}")
            print(f"  Y: {end_pose.position.y_val:.2f}")
            print(f"  Z: {end_pose.position.z_val:.2f}")
            
        except Exception as e:
            print(f"!!! ERROR during drone movement: {e}")
        
        print(f"=== END DRONE MOVEMENT ===\n")

    def _land_and_reset(self):
        """Perform simplified landing sequence and reset drone"""
        print("\nSimplified landing sequence...")
        print("Waiting 3 seconds before reset...")
        time.sleep(3)
        
        try:
            print("Resetting drone...")
            self.airsim_client.reset()
            print("Drone reset complete")
        except Exception as e:
            print(f"Error resetting drone: {e}")
        
        print("Landing and reset sequence complete")

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
        camera_config = self.camera_config[camera_id]
        
        # Extract camera position (handle both old and new schemas)
        location = {}
        if "location" in camera_config:
            location = camera_config["location"]
        elif "position" in camera_config:
            location = camera_config["position"]
        
        # Extract camera rotation (handle both old and new schemas)
        rotation = {}
        if "rotation" in camera_config:
            rotation = camera_config["rotation"]
        elif "orientation" in camera_config:
            rotation = camera_config["orientation"]
        
        # Extract camera specs (handle both old and new schemas)
        resolution = None
        fov = None
        
        if "resolution" in camera_config:
            resolution = camera_config["resolution"]
        elif "specs" in camera_config and "resolution" in camera_config["specs"]:
            resolution = camera_config["specs"]["resolution"]
        
        if "fov" in camera_config:
            fov = camera_config["fov"]
        elif "specs" in camera_config and "fov" in camera_config["specs"]:
            fov = camera_config["specs"]["fov"]
        
        # Convert camera_id to numeric ID (e.g., 'camera1' -> 1)
        camera_num = int(camera_id.replace('camera', ''))
        
        # Ensure simulation is running for camera setup
        if self.unrealcv_client.request('vget /action/game/is_paused') == 'true':
            self.unrealcv_client.request('vset /action/game/resume')
            print("Resumed simulation for camera setup")
        
        # Debug: Check available cameras before setup
        camera_list_before = self.unrealcv_client.request('vget /cameras')
        print(f"Available cameras before setup: {camera_list_before}")
        
        # Spawn and configure the camera
        print(f"Spawning camera {camera_num}...")
        spawn_result1 = self.unrealcv_client.request('vset /cameras/spawn')
        spawn_result2 = self.unrealcv_client.request('vset /cameras/spawn')
        print(f"Spawn results: {spawn_result1}, {spawn_result2}")
        
        # Set camera location
        location_cmd = f'vset /camera/{camera_num}/location {location["x"]} {location["y"]} {location["z"]}'
        location_result = self.unrealcv_client.request(location_cmd)
        print(f"Location command: {location_cmd}")
        print(f"Location result: {location_result}")
        
        # Set camera rotation
        rotation_cmd = f'vset /camera/{camera_num}/rotation {rotation["pitch"]} {rotation["yaw"]} {rotation["roll"]}'
        rotation_result = self.unrealcv_client.request(rotation_cmd)
        print(f"Rotation command: {rotation_cmd}")
        print(f"Rotation result: {rotation_result}")
        
        # Set camera resolution if specified in config
        if resolution:
            resolution_cmd = f'vset /camera/{camera_num}/size {resolution["width"]} {resolution["height"]}'
            resolution_result = self.unrealcv_client.request(resolution_cmd)
            print(f"Resolution command: {resolution_cmd}")
            print(f"Resolution result: {resolution_result}")
        
        # Set camera FOV if specified in config using the new method
        if fov:
            # Use the new set_cam_fov method
            self.set_cam_fov(camera_num, fov)
        
        # Debug: Check available cameras after setup
        camera_list_after = self.unrealcv_client.request('vget /cameras')
        print(f"Available cameras after setup: {camera_list_after}")
        
        print(f"{camera_id} set to location {location}, rotation {rotation}, "
              f"resolution {resolution}, FOV {fov}")

    def _execute_mission(self, camera_id):
        """Execute the drone mission and capture images sequentially at each waypoint."""
        # We need AirSim client to move the drone regardless of state saving
        if not self.airsim_client:
            print("ERROR: No AirSim client available. Cannot execute drone mission.")
            print("Please ensure AirSim is running and correctly configured.")
            time.sleep(15)  # Wait a bit to acknowledge the error
            return
            
        mission_file = self.config['drone_config']['mission_file']
        
        # Print some debug information about the mission file
        print(f"\n=== MISSION DETAILS FOR {camera_id} ===")
        print(f"Mission file: {mission_file}")
        try:
            if not os.path.exists(mission_file):
                print(f"WARNING: Mission file not found at {mission_file}")
                alternative_path = os.path.join(os.path.dirname(__file__), mission_file)
                print(f"Trying alternative path: {alternative_path}")
                if os.path.exists(alternative_path):
                    mission_file = alternative_path
                    print(f"Found mission file at alternative path")
                else:
                    print(f"ERROR: Could not find mission file at any location")
                    return
        except Exception as e:
            print(f"Error checking mission file: {e}")
            
        with open(mission_file, 'r') as f:
            mission_data = json.load(f)
            print(f"Successfully loaded mission data from {mission_file}")
        
        # Check if the new JSON schema is used (with "mission" wrapper)
        if "mission" in mission_data:
            print(f"Detected new mission schema with 'mission' wrapper")
            mission_data = mission_data["mission"]
        else:
            print(f"Using legacy mission schema format")
        
        target_drone = mission_data['drones'][0] # Simplified for now
        waypoints = target_drone['waypoints']
        print(f"Found {len(waypoints)} waypoints for drone")
        print(f"=== END MISSION DETAILS ===\n")
        
        # Get camera number once
        camera_num = int(camera_id.replace('camera', ''))
        max_images = self.config['data_collection'].get('max_images', float('inf'))
        start_time = time.time()
        
        # Navigate through waypoints
        for i, waypoint in enumerate(waypoints):
            # Check if max images or duration reached before moving
            if self.frame_index >= max_images or (time.time() - start_time) >= self.capture_duration:
                print("Reached max images or duration limit. Ending mission.")
                break
                
            print(f"\n>>> NAVIGATING TO WAYPOINT {waypoint['number']} ({i+1}/{len(waypoints)}) <<< ({camera_id})")
            
            # Use waypoint speed if specified, otherwise default to 5 m/s
            velocity = waypoint.get('speed', 5)
            
            # Print waypoint details
            print(f"Waypoint details:")
            print(json.dumps(waypoint, indent=2))
            
            # --- Move to Waypoint --- 
            # For local coordinate waypoints (assuming this based on previous context)
            x = waypoint.get('x', 0)
            y = waypoint.get('y', 0)
            altitude = waypoint.get('altitude', 0)
            
            # Check if position is nested (new format)
            if 'position' in waypoint:
                if 'x' in waypoint['position']: x = waypoint['position']['x']
                if 'y' in waypoint['position']: y = waypoint['position']['y']
                if 'z' in waypoint['position']: altitude = waypoint['position']['z'] # Assuming z is altitude
            
            print(f"Using local coordinates: X={x}, Y={y}, Alt={altitude}")
            # Call the modified _move_to_waypoint
            self._move_to_waypoint(
                x=x,
                y=y,
                altitude=altitude, # Pass altitude directly
                velocity=velocity
            )
            # Note: _move_to_waypoint now includes the .join()
            
            print(f"Arrived at waypoint {waypoint['number']}. Waiting 3 seconds...")
            time.sleep(1) 
            
            # --- Pause, Capture, Resume --- 
            print(f"Pausing simulation for image capture ({camera_id}, Frame: {self.frame_index})...")
            self.unrealcv_client.request('vset /action/game/pause')
            time.sleep(0.5) # Short delay to ensure pause takes effect
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{self.frame_index}_{timestamp}_{camera_id}"
            
            # Get appropriate capture directories (subset-specific if available)
            capture_dirs = self._get_camera_capture_dirs(camera_id)
            
            # Capture Mask
            mask_filename = f"{base_filename}_object_mask.png"
            mask_path = os.path.join(capture_dirs['mask'], mask_filename)
            mask_result = self.unrealcv_client.request(f'vget /camera/{camera_num}/object_mask {mask_path}')
            print(f"  Mask capture ({mask_filename}): {mask_result}")
            
            # Capture RGB
            rgb_filename = f"{base_filename}_lit.png"
            rgb_path = os.path.join(capture_dirs['rgb'], rgb_filename)
            rgb_result = self.unrealcv_client.request(f'vget /camera/{camera_num}/lit {rgb_path}')
            print(f"  RGB capture ({rgb_filename}): {rgb_result}")
            
            # Save State if enabled
            if self.save_state and capture_dirs['state']:
                print("  Saving state data...")
                self._save_state_to_json(camera_id, base_filename, timestamp, self.frame_index)
                print("  State data saved.")
                
            self.frame_index += 1
            
            print(f"Resuming simulation...")
            self.unrealcv_client.request('vset /action/game/resume')
            time.sleep(0.5) # Short delay after resuming
            # --- End Capture --- 

            print(f"Completed processing for waypoint {waypoint['number']}")
            # time.sleep(1) # Removed redundant sleep, already have wait + capture time
        
        print(f"\nCompleted all waypoints for {camera_id}")
        # Print final stats using the actual capture directories
        capture_dirs = self._get_camera_capture_dirs(camera_id)
        print(f"Captured {self.frame_index} frames for {camera_id}")
        print(f"{camera_id} images saved to:")
        print(f"  RGB: {capture_dirs['rgb']}")
        print(f"  Mask: {capture_dirs['mask']}")
        if self.save_state and capture_dirs['state']:
            print(f"  State: {capture_dirs['state']}")

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

    def _get_camera_fov(self, camera_num):
        """Get the FOV of a camera from UnrealCV"""
        try:
            fov_result = self.unrealcv_client.request(f'vget /camera/{camera_num}/fov')
            return float(fov_result)
        except Exception as e:
            print(f"Error getting FOV for camera {camera_num}: {e}")
            return 90.0  # Default FOV
            
    def _get_camera_position_rotation(self, camera_num):
        """Get camera position and rotation from UnrealCV"""
        try:
            position_result = self.unrealcv_client.request(f'vget /camera/{camera_num}/location')
            rotation_result = self.unrealcv_client.request(f'vget /camera/{camera_num}/rotation')
            
            # Parse position (format: X Y Z)
            position_parts = position_result.split(' ')
            position = {
                'x': float(position_parts[0]),
                'y': float(position_parts[1]),
                'z': float(position_parts[2])
            }
            
            # Parse rotation (format: Pitch Yaw Roll)
            rotation_parts = rotation_result.split(' ')
            rotation = {
                'pitch': float(rotation_parts[0]),
                'yaw': float(rotation_parts[1]),
                'roll': float(rotation_parts[2])
            }
            
            return position, rotation
        except Exception as e:
            print(f"Error getting position/rotation for camera {camera_num}: {e}")
            return {'x': 0, 'y': 0, 'z': 0}, {'pitch': 0, 'yaw': 0, 'roll': 0}
            
    def _execute_airsim_operation(self, operation_name, operation_func):
        """Execute an AirSim operation in a thread-safe way using a temporary client"""
        
        # Define a worker function that will run in a separate thread
        def worker():
            temp_client = None
            try:
                # Create and connect a temporary client for this operation
                temp_client = MultirotorClient()
                temp_client.confirmConnection() # Ensure connection before proceeding
                
                # Execute the passed operation function using the temporary client
                result = operation_func(temp_client)
                self.airsim_queue.put(result)
            except Exception as e:
                # Print specific error for context
                print(f"Error in AirSim operation '{operation_name}' using temp client: {e}")
                self.airsim_queue.put(None) # Signal failure
        
        # Create and start a thread for the operation
        thread = threading.Thread(target=worker)
        thread.daemon = True # Allow program to exit even if this thread hangs
        thread.start()
        
        # Wait for the result with a timeout
        try:
            # Increased timeout slightly for connection overhead
            return self.airsim_queue.get(timeout=3.0) 
        except queue.Empty:
            print(f"Timeout waiting for AirSim operation '{operation_name}'")
            return None
            
    def _get_drone_state(self):
        """Get current drone state from AirSim including ground truth environment data"""
        # Only print verbose state info if state saving is enabled
        if self.save_state:
            print("Getting drone state...")
        
        try:
            # --- Direct method without using the thread-safe queue ---
            if not self.airsim_client:
                print("Warning: No AirSim client available for getting drone state")
                return self._get_default_drone_state()
                
            # --- Get Multirotor State --- 
            try:
                drone_state_data = self.airsim_client.getMultirotorState()
                if drone_state_data:
                    position = {
                        'x': float(drone_state_data.kinematics_estimated.position.x_val),
                        'y': float(drone_state_data.kinematics_estimated.position.y_val),
                        'z': float(drone_state_data.kinematics_estimated.position.z_val)
                    }
                    orientation = {
                        'w': float(drone_state_data.kinematics_estimated.orientation.w_val),
                        'x': float(drone_state_data.kinematics_estimated.orientation.x_val),
                        'y': float(drone_state_data.kinematics_estimated.orientation.y_val),
                        'z': float(drone_state_data.kinematics_estimated.orientation.z_val)
                    }
                    # Only print verbose state info if state saving is enabled
                    if self.save_state:
                        print(f"Got drone position: X={position['x']:.2f}, Y={position['y']:.2f}, Z={position['z']:.2f}")
                else:
                    print("Warning: getMultirotorState returned None")
                    position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    orientation = {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
            except Exception as e:
                print(f"Error getting multirotor state: {e}")
                position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                orientation = {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}

            # --- Get GPS Data --- 
            try:
                gps_data = self.airsim_client.getGpsData(gps_name="", vehicle_name="")
                if gps_data:
                    gps_info = {
                        'latitude': float(gps_data.gnss.geo_point.latitude),
                        'longitude': float(gps_data.gnss.geo_point.longitude),
                        'altitude': float(gps_data.gnss.geo_point.altitude)
                    }
                    # Only print verbose state info if state saving is enabled
                    if self.save_state:
                        print(f"Got GPS data: Lat={gps_info['latitude']:.6f}, Lon={gps_info['longitude']:.6f}, Alt={gps_info['altitude']:.2f}")
                else:
                    print("Warning: getGpsData returned None")
                    gps_info = {'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0}
            except Exception as e:
                print(f"Error getting GPS data: {e}")
                gps_info = {'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0}
            
            # --- Get Environment Data --- 
            try:
                env_data = self.airsim_client.simGetGroundTruthEnvironment()
                if env_data:
                    environment = {
                        'position': {
                            'x': float(env_data.position.x_val),
                            'y': float(env_data.position.y_val),
                            'z': float(env_data.position.z_val)
                        },
                        'geo_point': {
                            'latitude': float(env_data.geo_point.latitude),
                            'longitude': float(env_data.geo_point.longitude),
                            'altitude': float(env_data.geo_point.altitude)
                        },
                        'gravity': {
                            'x': float(env_data.gravity.x_val),
                            'y': float(env_data.gravity.y_val),
                            'z': float(env_data.gravity.z_val)
                        },
                        'air_pressure': float(env_data.air_pressure),
                        'temperature': float(env_data.temperature),
                        'air_density': float(env_data.air_density)
                    }
                    # Only print verbose state info if state saving is enabled
                    if self.save_state:
                        print(f"Got environment data")
                else:
                    print("Warning: simGetGroundTruthEnvironment returned None")
                    environment = self._get_default_environment()
            except Exception as e:
                print(f"Error getting environment data: {e}")
                environment = self._get_default_environment()
                
            # --- Check if we got any valid data ---
            if position['x'] == 0.0 and position['y'] == 0.0 and position['z'] == 0.0 and \
               gps_info['latitude'] == 0.0 and gps_info['longitude'] == 0.0:
                print("Warning: All position data is zeros. This might indicate a connection issue.")
                
            # --- Combine Results --- 
            drone_state = {
                'position': position,
                'orientation': orientation,
                'gps': gps_info,
                'environment': environment
            }
            
            return drone_state
            
        except Exception as e:
            # Catch any unexpected error during the assembly process
            print(f"Unexpected error assembling drone state: {e}")
            return self._get_default_drone_state()

    def _save_state_to_json(self, camera_id, base_filename, timestamp, frame_index):
        """Save camera and drone state to a JSON file"""
        # Skip if state saving is disabled
        if not self.save_state:
            return
            
        camera_num = int(camera_id.replace('camera', ''))
        
        # Get camera position, rotation and FOV
        camera_position, camera_rotation = self._get_camera_position_rotation(camera_num)
        camera_fov = self._get_camera_fov(camera_num)
        
        # Get drone state from AirSim
        drone_state = self._get_drone_state()
        
        # Prepare state data
        state_data = {
            'timestamp': timestamp,
            'frame_index': frame_index,
            'camera': {
                'id': camera_id,
                'position': camera_position,
                'rotation': camera_rotation,
                'fov': camera_fov
            },
            'drone': drone_state
        }
        
        # Get appropriate capture directories (subset-specific if available)
        capture_dirs = self._get_camera_capture_dirs(camera_id)
        
        # Create state filename to match image filename pattern
        state_filename = f"{frame_index}_{timestamp}_{camera_id}_state.json"
        state_path = os.path.join(capture_dirs['state'], state_filename)
        
        # Save to JSON file
        with open(state_path, 'w') as f:
            json.dump(state_data, f, indent=2)

    def _setup_all_cameras(self):
        """Set up all cameras at once (used in parallel mode)"""
        print("\n=== Setting up all cameras for parallel collection ===")
        for camera_id in self.camera_config.keys():
            self._setup_specific_camera(camera_id)
        print("=== All cameras set up successfully ===\n")

    def _execute_mission_parallel(self):
        """Execute the drone mission and capture from all cameras at each waypoint."""
        # We need AirSim client to move the drone
        if not self.airsim_client:
            print("ERROR: No AirSim client available. Cannot execute drone mission.")
            print("Please ensure AirSim is running and correctly configured.")
            time.sleep(15)  # Wait a bit to acknowledge the error
            return
            
        mission_file = self.config['drone_config']['mission_file']
        
        # Print some debug information about the mission file
        print(f"\n=== MISSION DETAILS FOR PARALLEL MODE ===")
        try:
            if not os.path.exists(mission_file):
                print(f"WARNING: Mission file not found at {mission_file}")
                alternative_path = os.path.join(os.path.dirname(__file__), mission_file)
                print(f"Trying alternative path: {alternative_path}")
                if os.path.exists(alternative_path):
                    mission_file = alternative_path
                    print(f"Found mission file at alternative path")
                else:
                    print(f"ERROR: Could not find mission file at any location")
                    return
        except Exception as e:
            print(f"Error checking mission file: {e}")
            
        with open(mission_file, 'r') as f:
            mission_data = json.load(f)
            print(f"Successfully loaded mission data from {mission_file}")
        
        # Check if the new JSON schema is used (with "mission" wrapper)
        if "mission" in mission_data:
            print(f"Detected new mission schema with 'mission' wrapper")
            mission_data = mission_data["mission"]
        else:
            print(f"Using legacy mission schema format")
        
        target_drone = mission_data['drones'][0] # Simplified for now
        waypoints = target_drone['waypoints']
        print(f"Found {len(waypoints)} waypoints for drone")
        print(f"=== END MISSION DETAILS ===\n")
        
        # Get all camera numbers once
        camera_nums = {camera_id: int(camera_id.replace('camera', '')) 
                      for camera_id in self.camera_config.keys()}
        
        max_images = self.config['data_collection'].get('max_images', float('inf'))
        start_time = time.time()
        
        # Dictionary to track frame index for each camera
        frame_indices = {camera_id: self._get_max_frame_index_for_camera(camera_id) 
                        for camera_id in self.camera_config.keys()}
        
        # Navigate through waypoints
        for i, waypoint in enumerate(waypoints):
            # Check if max images or duration reached before moving
            any_camera_max_reached = any(index >= max_images for index in frame_indices.values())
            if any_camera_max_reached or (time.time() - start_time) >= self.capture_duration:
                print("Reached max images or duration limit. Ending mission.")
                break
                
            print(f"\n>>> NAVIGATING TO WAYPOINT {waypoint['number']} ({i+1}/{len(waypoints)}) <<< (PARALLEL MODE)")
            
            velocity = waypoint.get('speed', 5)
            
            # Print waypoint details
            print(f"Waypoint details:")
            print(json.dumps(waypoint, indent=2))
            
            # Extract waypoint coordinates as in _execute_mission
            x = waypoint.get('x', 0)
            y = waypoint.get('y', 0)
            altitude = waypoint.get('altitude', 0)
            
            # Check if position is nested (new format)
            if 'position' in waypoint:
                if 'x' in waypoint['position']: x = waypoint['position']['x']
                if 'y' in waypoint['position']: y = waypoint['position']['y']
                if 'z' in waypoint['position']: altitude = waypoint['position']['z']
            
            print(f"Using local coordinates: X={x}, Y={y}, Alt={altitude}")
            # Move drone to waypoint
            self._move_to_waypoint(
                x=x,
                y=y,
                altitude=altitude,
                velocity=velocity
            )
            
            print(f"Arrived at waypoint {waypoint['number']}. Waiting 3 seconds...")
            time.sleep(3)
            
            # --- Capture from all cameras sequentially ---
            print(f"Pausing simulation for image capture at waypoint {waypoint['number']}...")
            self.unrealcv_client.request('vset /action/game/pause')
            time.sleep(0.5)  # Short delay to ensure pause takes effect
            
            for camera_id, camera_num in camera_nums.items():
                # Skip if this camera has reached max images
                if frame_indices[camera_id] >= max_images:
                    print(f"Skipping {camera_id} - reached max images")
                    continue
                    
                print(f"Capturing from {camera_id} (frame {frame_indices[camera_id]})...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"{frame_indices[camera_id]}_{timestamp}_{camera_id}"
                
                # Get appropriate capture directories (subset-specific if available)
                capture_dirs = self._get_camera_capture_dirs(camera_id)
                
                if not capture_dirs['rgb'] or not capture_dirs['mask']:
                    print(f"Error: Invalid capture directories for {camera_id}")
                    print(f"RGB: {capture_dirs['rgb']}, Mask: {capture_dirs['mask']}")
                    continue
                
                print(f"  Capture directories for {camera_id}:")
                print(f"    RGB dir: {capture_dirs['rgb']}")
                print(f"    Mask dir: {capture_dirs['mask']}")
                
                # Capture Mask
                mask_filename = f"{base_filename}_object_mask.png"
                mask_path = os.path.join(capture_dirs['mask'], mask_filename)
                print(f"  Attempting mask capture to: {mask_path}")
                mask_result = self.unrealcv_client.request(f'vget /camera/{camera_num}/object_mask {mask_path}')
                print(f"  Mask capture result: {mask_result}")
                
                # Verify mask file was created
                if os.path.exists(mask_path):
                    file_size = os.path.getsize(mask_path)
                    print(f"  ✓ Mask file created: {mask_path} ({file_size} bytes)")
                else:
                    print(f"  ✗ Mask file NOT created: {mask_path}")
                
                # Capture RGB
                rgb_filename = f"{base_filename}_lit.png"
                rgb_path = os.path.join(capture_dirs['rgb'], rgb_filename)
                print(f"  Attempting RGB capture to: {rgb_path}")
                rgb_result = self.unrealcv_client.request(f'vget /camera/{camera_num}/lit {rgb_path}')
                print(f"  RGB capture result: {rgb_result}")
                
                # Verify RGB file was created
                if os.path.exists(rgb_path):
                    file_size = os.path.getsize(rgb_path)
                    print(f"  ✓ RGB file created: {rgb_path} ({file_size} bytes)")
                else:
                    print(f"  ✗ RGB file NOT created: {rgb_path}")
                
                # Save State if enabled
                if self.save_state and capture_dirs['state']:
                    print(f"  Saving state data for {camera_id}...")
                    self._save_state_to_json(camera_id, base_filename, timestamp, frame_indices[camera_id])
                
                # Increment frame index for this camera
                frame_indices[camera_id] += 1
            
            print(f"Resuming simulation...")
            self.unrealcv_client.request('vset /action/game/resume')
            time.sleep(0.5)  # Short delay after resuming
            
            print(f"Completed processing for waypoint {waypoint['number']}")
        
        print(f"\nCompleted all waypoints in parallel mode")
        # Print final stats for each camera using actual capture directories
        print("Capture statistics:")
        for camera_id, frame_count in frame_indices.items():
            capture_dirs = self._get_camera_capture_dirs(camera_id)
            print(f"  {camera_id}: {frame_count} frames")
            print(f"    RGB: {capture_dirs['rgb']}")
            print(f"    Mask: {capture_dirs['mask']}")
            if self.save_state and capture_dirs['state']:
                print(f"    State: {capture_dirs['state']}")

    def _extract_camera_fov(self, camera_config):
        """
        Extract camera FOV from various possible locations in camera config.
        
        Parameters:
            camera_config (dict): The camera configuration dictionary
            
        Returns:
            float or None: The camera FOV in degrees or None if not specified
        """
        # Check various possible locations for FOV settings
        if "fov" in camera_config:
            return float(camera_config["fov"])
        elif "specs" in camera_config and "fov" in camera_config["specs"]:
            return float(camera_config["specs"]["fov"])
        # Check if we have a default FOV in the main config
        elif "default_fov" in self.config.get("data_collection", {}):
            return float(self.config["data_collection"]["default_fov"])
        return None
        
    def _setup_cameras_with_fov(self):
        """Set up all cameras with the FOV values from the config file"""
        print("\n=== Setting up all cameras with specified FOVs ===")
        for camera_id, camera_config in self.camera_config.items():
            try:
                camera_num = int(camera_id.replace('camera', ''))
                # Extract FOV value
                fov = self._extract_camera_fov(camera_config)
                
                if fov is not None:
                    self.set_cam_fov(camera_num, fov)
                    print(f"Set {camera_id} FOV to {fov} degrees")
                else:
                    print(f"No FOV specified for {camera_id}, using default")
            except Exception as e:
                print(f"Error setting up FOV for {camera_id}: {e}")
        print("=== Camera FOV setup complete ===\n")

    def run(self):
        """
        Run the data collection process based on the selected mode:
        - Sequential mode: Process each camera fully before moving to the next
        - Parallel mode: Set up all cameras and capture from all at each waypoint
        - Subset mode: When z_folder is provided, iterate through subset configurations
        - Launch-all-parallel mode: Set up all cameras from all subsets and capture from all simultaneously
        """
        try:
            # Check if we're using launch-all-parallel mode
            if self.launch_all_parallel and self.z_folder and self.subset_files:
                print(f"\n=== Starting LAUNCH-ALL-PARALLEL data collection ===")
                print(f"Processing all {len(self.subset_files)} subsets simultaneously")
                
                # Set up initial agent colors (only needs to be done once)
                self._setup_camera()
                
                # Load and setup all cameras from all subsets
                self._load_all_subset_cameras()
                self._setup_all_subset_cameras()
                
                # Initialize drone and take off once for the whole mission
                if self.airsim_client:
                    try:
                        self._initialize_drone()
                        self._takeoff()
                        
                        # Print initial position after takeoff
                        if self.config['drone_config'].get('coordinate_system') == 'geographic':
                            self._print_current_altitude()
                        else:
                            self._print_current_position()
                            
                        # Execute mission capturing from ALL subset cameras
                        self._execute_mission_launch_all_parallel()
                        
                        # Land drone after completing all waypoints
                        self._land_and_reset()
                    except Exception as e:
                        print(f"Error during launch-all-parallel mission execution: {e}")
                        try:
                            if self.airsim_client:
                                print("Attempting emergency landing after error...")
                                self._land_and_reset()
                        except Exception as land_e:
                            print(f"Error during emergency landing: {land_e}")
                else:
                    print("WARNING: No AirSim client available. Cannot control drone.")
                
                print("=== Completed LAUNCH-ALL-PARALLEL data collection ===")
                return
            
            # Check if we're using subset files from z_folder (but not launch-all-parallel)
            elif self.z_folder and self.subset_files:
                print(f"\n=== Starting data collection with {len(self.subset_files)} subset configurations ===")
                
                # Iterate through each subset file
                for subset_index, subset_file in enumerate(self.subset_files):
                    print(f"\n--- Processing subset {subset_index + 1}/{len(self.subset_files)}: {os.path.basename(subset_file)} ---")
                    
                    # Load subset configuration
                    try:
                        with open(subset_file, 'r') as f:
                            subset_config = json.load(f)
                        self._set_subset_config(subset_config)
                        
                        # Set up initial agent colors (only needs to be done once per subset)
                        self._setup_camera()
                        
                        # Setup cameras based on subset configuration
                        active_cameras = self._setup_cameras_from_subset(subset_config)
                        
                        # Create subset-specific directories
                        self._setup_subset_directories(subset_file, active_cameras)
                        
                        if self.parallel_mode:
                            self._run_subset_parallel_mode(active_cameras)
                        else:
                            self._run_subset_sequential_mode(active_cameras)
                            
                    except Exception as e:
                        print(f"Error processing subset {subset_file}: {e}")
                        continue
                        
                print("=== Completed all subset configurations ===")
                return
                
            # Original behavior when no z_folder is provided
            else:
                self._run_original_mode()
            
        except Exception as e:
            print(f"Error in main run method: {e}")

    def _run_original_mode(self):
        """Run the original data collection mode without subset files"""
        # Set up initial agent colors (only needs to be done once)
        self._setup_camera()
        
        # Setup all cameras with the FOV values from the config file
        self._setup_cameras_with_fov()
        
        if self.parallel_mode:
            # --- PARALLEL MODE ---
            print("\n=== Starting data collection in PARALLEL mode ===")
            
            # Set up all cameras at once
            self._setup_all_cameras()
            
            # Initialize drone and take off once for the whole mission
            if self.airsim_client:
                try:
                    self._initialize_drone()
                    self._takeoff()
                    
                    # Print initial position after takeoff
                    if self.config['drone_config'].get('coordinate_system') == 'geographic':
                        self._print_current_altitude()
                    else:
                        self._print_current_position()
                        
                    # Execute mission in parallel mode (all cameras capture at each waypoint)
                    self._execute_mission_parallel()
                    
                    # Land drone after completing all waypoints
                    self._land_and_reset()
                except Exception as e:
                    print(f"Error during parallel mission execution: {e}")
                    try:
                        if self.airsim_client:
                            print("Attempting emergency landing after error...")
                            self._land_and_reset()
                    except Exception as land_e:
                        print(f"Error during emergency landing: {land_e}")
            else:
                print("WARNING: No AirSim client available. Cannot control drone.")
            
            print("=== Completed data collection in PARALLEL mode ===\n")
            
        else:
            # --- SEQUENTIAL MODE ---
            # Process each camera in full sequence (existing code)
            for camera_id in self.camera_config.keys():
                print(f"\n=== Starting complete data collection cycle for {camera_id} ===")
                
                # Set up this specific camera's position/rotation
                self._setup_specific_camera(camera_id)
                
                # Reset frame index based on existing files for this camera's output dirs
                self.frame_index = self._get_max_frame_index_for_camera(camera_id)
                print(f"Starting data collection for {camera_id} from frame index: {self.frame_index}")
                
                # Initialize drone and take off - Do this once for this camera's full cycle
                if self.airsim_client:
                    try:
                        self._initialize_drone()
                        self._takeoff()
                        
                        # Print initial position after takeoff
                        if self.config['drone_config'].get('coordinate_system') == 'geographic':
                            self._print_current_altitude()
                        else:
                            self._print_current_position()
                            
                        # Execute mission (which now includes sequential capture)
                        self._execute_mission(camera_id)
                        
                        # Land drone after completing all waypoints for this camera
                        self._land_and_reset()
                    except Exception as e:
                        print(f"Error during mission execution for {camera_id}: {e}")
                        # Try to land/reset even if there was an error
                        try:
                            if self.airsim_client:
                                print("Attempting emergency landing after error...")
                                self._land_and_reset()
                        except Exception as land_e:
                            print(f"Error during emergency landing: {land_e}")
                else:
                    print("WARNING: No AirSim client available. Cannot control drone.")
                
                print(f"=== Completed full data collection cycle for {camera_id} ===\n")
                
                # Pause between camera cycles to ensure everything is cleaned up
                print(f"Pausing for 5 seconds before starting next camera...")
                time.sleep(5)
            
            # Final completion message
            print("\n=== All camera cycles completed successfully ===")
        
        # Handle any errors and cleanup
        try:
            # Any final cleanup if needed
            pass
        except Exception as e:
            print(f"Error during data collection run: {e}")
        finally:
            try:
                # Final cleanup if needed
                if self.airsim_client and hasattr(self, 'initial_position') and self.initial_position:
                    print("Performing final landing and reset...")
                    # Make sure simulation is running before landing/resetting
                    if self.unrealcv_client.request('vget /action/game/is_paused') == 'true':
                        self.unrealcv_client.request('vset /action/game/resume')
                    self._land_and_reset()
            except Exception as e:
                print(f"Error during final landing/reset: {e}")
            
            if self.unrealcv_client.isconnected():
                self.unrealcv_client.disconnect()
            print("Data collection complete.")
            return

    def _run_subset_parallel_mode(self, active_cameras):
        """Run data collection in parallel mode for subset configuration"""
        print("\n=== Starting subset data collection in PARALLEL mode ===")
        
        # Initialize drone and take off once for the whole mission
        if self.airsim_client:
            try:
                self._initialize_drone()
                self._takeoff()
                
                # Print initial position after takeoff
                if self.config['drone_config'].get('coordinate_system') == 'geographic':
                    self._print_current_altitude()
                else:
                    self._print_current_position()
                    
                # Execute mission in parallel mode (all active cameras capture at each waypoint)
                self._execute_mission_parallel_subset(active_cameras)
                
                # Land drone after completing all waypoints
                self._land_and_reset()
            except Exception as e:
                print(f"Error during subset parallel mission execution: {e}")
                try:
                    if self.airsim_client:
                        print("Attempting emergency landing after error...")
                        self._land_and_reset()
                except Exception as land_e:
                    print(f"Error during emergency landing: {land_e}")
        else:
            print("WARNING: No AirSim client available. Cannot control drone.")
        
        print("=== Completed subset data collection in PARALLEL mode ===\n")
        
    def _run_subset_sequential_mode(self, active_cameras):
        """Run data collection in sequential mode for subset configuration"""
        print("\n=== Starting subset data collection in SEQUENTIAL mode ===")
        
        # Process each active camera in full sequence
        for cam_name, config in active_cameras.items():
            camera_id = config["camera_id"]
            print(f"\n=== Starting complete data collection cycle for {cam_name} (ID: {camera_id}) ===")
            
            # Reset frame index based on existing files for this camera's output dirs
            self.frame_index = self._get_max_frame_index_for_camera(camera_id)
            print(f"Starting data collection for {cam_name} from frame index: {self.frame_index}")
            
            # Initialize drone and take off - Do this once for this camera's full cycle
            if self.airsim_client:
                try:
                    self._initialize_drone()
                    self._takeoff()
                    
                    # Print initial position after takeoff
                    if self.config['drone_config'].get('coordinate_system') == 'geographic':
                        self._print_current_altitude()
                    else:
                        self._print_current_position()
                        
                    # Execute mission for this camera
                    self._execute_mission_subset(camera_id)
                    
                    # Land drone after completing all waypoints for this camera
                    self._land_and_reset()
                except Exception as e:
                    print(f"Error during mission execution for {cam_name}: {e}")
                    # Try to land/reset even if there was an error
                    try:
                        if self.airsim_client:
                            print("Attempting emergency landing after error...")
                            self._land_and_reset()
                    except Exception as land_e:
                        print(f"Error during emergency landing: {land_e}")
            else:
                print("WARNING: No AirSim client available. Cannot control drone.")
            
            print(f"=== Completed full data collection cycle for {cam_name} ===\n")
            
            # Pause between camera cycles to ensure everything is cleaned up
            print(f"Pausing for 5 seconds before starting next camera...")
            time.sleep(5)
        
        print("\n=== All subset camera cycles completed successfully ===")
        
    def _execute_mission_parallel_subset(self, active_cameras):
        """Execute mission in parallel mode for subset - all active cameras capture at each waypoint"""
        print("Starting parallel mission execution for subset...")
        
        # Get camera IDs from active cameras
        camera_ids = [str(config["camera_id"]) for config in active_cameras.values()]
        print(f"Active cameras for parallel execution: {camera_ids}")
        
        # We need AirSim client to move the drone
        if not self.airsim_client:
            print("ERROR: No AirSim client available. Cannot execute drone mission.")
            print("Please ensure AirSim is running and correctly configured.")
            time.sleep(15)  # Wait a bit to acknowledge the error
            return
            
        mission_file = self.config['drone_config']['mission_file']
        
        # Print some debug information about the mission file
        print(f"\n=== MISSION DETAILS FOR SUBSET PARALLEL MODE ===")
        try:
            if not os.path.exists(mission_file):
                print(f"WARNING: Mission file not found at {mission_file}")
                alternative_path = os.path.join(os.path.dirname(__file__), mission_file)
                print(f"Trying alternative path: {alternative_path}")
                if os.path.exists(alternative_path):
                    mission_file = alternative_path
                    print(f"Found mission file at alternative path")
                else:
                    print(f"ERROR: Could not find mission file at any location")
                    return
        except Exception as e:
            print(f"Error checking mission file: {e}")
            
        with open(mission_file, 'r') as f:
            mission_data = json.load(f)
            print(f"Successfully loaded mission data from {mission_file}")
        
        # Check if the new JSON schema is used (with "mission" wrapper)
        if "mission" in mission_data:
            print(f"Detected new mission schema with 'mission' wrapper")
            mission_data = mission_data["mission"]
        else:
            print(f"Using legacy mission schema format")
        
        target_drone = mission_data['drones'][0] # Simplified for now
        waypoints = target_drone['waypoints']
        print(f"Found {len(waypoints)} waypoints for drone")
        print(f"=== END MISSION DETAILS ===\n")
        
        # Create camera number mapping from active cameras only
        camera_nums = {}
        for cam_name, config in active_cameras.items():
            camera_id = config["camera_id"]
            camera_key = f"camera{camera_id}"
            camera_nums[camera_key] = camera_id
        
        print(f"Camera mapping for subset: {camera_nums}")
        
        max_images = self.config['data_collection'].get('max_images', float('inf'))
        start_time = time.time()
        
        # Dictionary to track frame index for each active camera
        frame_indices = {}
        for camera_key in camera_nums.keys():
            if camera_key in self.camera_dirs:
                frame_indices[camera_key] = self._get_max_frame_index_for_camera(camera_key)
            else:
                frame_indices[camera_key] = 0
                print(f"Warning: {camera_key} not found in camera_dirs, starting from frame 0")
        
        # Navigate through waypoints
        for i, waypoint in enumerate(waypoints):
            # Check if max images or duration reached before moving
            any_camera_max_reached = any(index >= max_images for index in frame_indices.values())
            if any_camera_max_reached or (time.time() - start_time) >= self.capture_duration:
                print("Reached max images or duration limit. Ending mission.")
                break
                
            print(f"\n>>> NAVIGATING TO WAYPOINT {waypoint['number']} ({i+1}/{len(waypoints)}) <<< (SUBSET PARALLEL MODE)")
            
            velocity = waypoint.get('speed', 5)
            
            # Print waypoint details
            print(f"Waypoint details:")
            print(json.dumps(waypoint, indent=2))
            
            # Extract waypoint coordinates
            x = waypoint.get('x', 0)
            y = waypoint.get('y', 0)
            altitude = waypoint.get('altitude', 0)
            
            # Check if position is nested (new format)
            if 'position' in waypoint:
                if 'x' in waypoint['position']: x = waypoint['position']['x']
                if 'y' in waypoint['position']: y = waypoint['position']['y']
                if 'z' in waypoint['position']: altitude = waypoint['position']['z']
            
            print(f"Using local coordinates: X={x}, Y={y}, Alt={altitude}")
            # Move drone to waypoint
            self._move_to_waypoint(
                x=x,
                y=y,
                altitude=altitude,
                velocity=velocity
            )
            
            print(f"Arrived at waypoint {waypoint['number']}. Waiting 3 seconds...")
            time.sleep(3)
            
            # --- Capture from all active cameras sequentially ---
            print(f"Pausing simulation for image capture at waypoint {waypoint['number']}...")
            self.unrealcv_client.request('vset /action/game/pause')
            time.sleep(0.5)  # Short delay to ensure pause takes effect
            
            for camera_key, camera_num in camera_nums.items():
                # Skip if this camera has reached max images
                if frame_indices[camera_key] >= max_images:
                    print(f"Skipping {camera_key} - reached max images")
                    continue
                    
                print(f"Capturing from {camera_key} (frame {frame_indices[camera_key]})...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"{frame_indices[camera_key]}_{timestamp}_{camera_key}"
                
                # Get appropriate capture directories
                capture_dirs = self._get_camera_capture_dirs(camera_key)
                
                if not capture_dirs['rgb'] or not capture_dirs['mask']:
                    print(f"Error: Invalid capture directories for {camera_key}")
                    print(f"RGB: {capture_dirs['rgb']}, Mask: {capture_dirs['mask']}")
                    continue
                
                print(f"  Capture directories for {camera_key}:")
                print(f"    RGB dir: {capture_dirs['rgb']}")
                print(f"    Mask dir: {capture_dirs['mask']}")
                
                # Capture Mask
                mask_filename = f"{base_filename}_object_mask.png"
                mask_path = os.path.join(capture_dirs['mask'], mask_filename)
                print(f"  Attempting mask capture to: {mask_path}")
                mask_result = self.unrealcv_client.request(f'vget /camera/{camera_num}/object_mask {mask_path}')
                print(f"  Mask capture result: {mask_result}")
                
                # Verify mask file was created
                if os.path.exists(mask_path):
                    file_size = os.path.getsize(mask_path)
                    print(f"  ✓ Mask file created: {mask_path} ({file_size} bytes)")
                else:
                    print(f"  ✗ Mask file NOT created: {mask_path}")
                
                # Capture RGB
                rgb_filename = f"{base_filename}_lit.png"
                rgb_path = os.path.join(capture_dirs['rgb'], rgb_filename)
                print(f"  Attempting RGB capture to: {rgb_path}")
                rgb_result = self.unrealcv_client.request(f'vget /camera/{camera_num}/lit {rgb_path}')
                print(f"  RGB capture result: {rgb_result}")
                
                # Verify RGB file was created
                if os.path.exists(rgb_path):
                    file_size = os.path.getsize(rgb_path)
                    print(f"  ✓ RGB file created: {rgb_path} ({file_size} bytes)")
                else:
                    print(f"  ✗ RGB file NOT created: {rgb_path}")
                
                # Save State if enabled
                if self.save_state and capture_dirs['state']:
                    print(f"  Saving state data for {camera_key}...")
                    self._save_state_to_json(camera_key, base_filename, timestamp, frame_indices[camera_key])
                
                # Increment frame index for this camera
                frame_indices[camera_key] += 1
            
            print(f"Resuming simulation...")
            self.unrealcv_client.request('vset /action/game/resume')
            time.sleep(0.5)  # Short delay after resuming
            
            print(f"Completed processing for waypoint {waypoint['number']}")
        
        print(f"\nCompleted all waypoints in subset parallel mode")
        # Print final stats for each camera using actual capture directories
        print("Capture statistics:")
        for camera_key, frame_count in frame_indices.items():
            capture_dirs = self._get_camera_capture_dirs(camera_key)
            print(f"  {camera_key}: {frame_count} frames")
            print(f"    RGB: {capture_dirs['rgb']}")
            print(f"    Mask: {capture_dirs['mask']}")
            if self.save_state and capture_dirs['state']:
                print(f"    State: {capture_dirs['state']}")

    def _execute_mission_subset(self, camera_id):
        """Execute mission for a specific camera in subset mode"""
        print(f"Starting mission execution for camera {camera_id}...")
        
        # Use the existing mission execution logic for single camera
        # Make sure we use the proper camera key format
        if isinstance(camera_id, int):
            camera_key = f"camera{camera_id}"
        else:
            camera_key = camera_id
            
        self._execute_mission(camera_key)

    def init_drone_type(self, drone_type):
        """
        Update the AirSim settings.json file to use the specified drone type.
        
        Args:
            drone_type (str): The type of drone to use (e.g., 'DJIS900', 'DJI_Phantom')
        """
        print(f"\n=== INITIALIZING DRONE TYPE: {drone_type} ===")
        
        # Path to AirSim settings.json
        settings_path = os.path.expanduser("~/Documents/AirSim/settings.json")
        
        try:
            # Check if settings file exists
            if not os.path.exists(settings_path):
                print(f"WARNING: AirSim settings file not found at {settings_path}")
                
                # Try alternative paths
                alternative_paths = [
                    os.path.expanduser("~/Documents/AirSim/settings.json"),
                    os.path.expanduser("~/.ros/AirSim/settings.json"),
                    os.path.expanduser("~/AirSim/settings.json"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
                ]
                
                for alt_path in alternative_paths:
                    print(f"Checking alternative path: {alt_path}")
                    if os.path.exists(alt_path):
                        settings_path = alt_path
                        print(f"Found AirSim settings at: {settings_path}")
                        break
                else:
                    print("WARNING: AirSim settings file not found at any location")
                    print("AirSim drone type configuration skipped")
                    return
            
            print(f"Using AirSim settings file at: {settings_path}")
            
            # Read current settings
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            
            # Create a backup of the original settings
            backup_path = os.path.join(os.path.dirname(settings_path), "settings_backup.json")
            with open(backup_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            # Print the current drone settings
            print("Current AirSim settings:")
            if 'PawnPaths' in settings and 'DefaultQuadrotor' in settings['PawnPaths']:
                current_pawn_bp = settings['PawnPaths']['DefaultQuadrotor'].get('PawnBP', 'Not set')
                print(f"Current PawnBP: {current_pawn_bp}")
            else:
                print("No DefaultQuadrotor configuration found in settings")
            
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
        
        print("=== DRONE TYPE INITIALIZATION COMPLETE ===\n")

    def _get_default_drone_state(self):
        """Return a default drone state structure with zeros"""
        return {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
            'gps': {'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0},
            'environment': self._get_default_environment()
        }
        
    def _get_default_environment(self):
        """Return a default environment structure with sensible defaults"""
        return {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'geo_point': {'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0},
            'gravity': {'x': 0.0, 'y': 0.0, 'z': -9.8},
            'air_pressure': 101325.0,  # Standard atmospheric pressure in Pa
            'temperature': 15.0,       # Standard temperature in Celsius
            'air_density': 1.225       # Standard air density at sea level in kg/m³
        }

    def _get_max_frame_index_for_camera(self, camera_id):
        """Find the highest frame index from existing files for a specific camera."""
        max_index = -1
        camera_dirs = self.camera_dirs.get(camera_id)
        if not camera_dirs:
            print(f"Warning: No directory information found for {camera_id}")
            return 0
            
        for directory_key in ['rgb', 'mask']:
            directory = camera_dirs.get(directory_key)
            if not directory or not os.path.exists(directory):
                continue
                
            for filename in os.listdir(directory):
                # Match pattern like 123_YYYYMMDD_HHMMSS_camera1_...
                # Use re.escape to handle camera_id potentially containing special regex characters
                match = re.match(r'^(\d+)_\d{8}_\d{6}_' + re.escape(camera_id) + r'_', filename)
                if match:
                    index = int(match.group(1))
                    max_index = max(max_index, index)
    
        # Return next index to start from
        return max_index + 1 if max_index >= 0 else 0

    def _find_existing_camera_dirs(self):
        """Find all existing camera directories to avoid naming conflicts."""
        existing_dirs = set()
        
        try:
            # Check if base output directory exists
            if not os.path.exists(self.base_output_dir):
                return existing_dirs
                
            # Look through directories in the base output dir
            for dirname in os.listdir(self.base_output_dir):
                # Extract the camera number if it matches our pattern (e.g., "urban-clearcheck-cam1")
                match = re.search(rf'{re.escape(self.weather_base_pattern)}(\d+)$', dirname)
                if match:
                    cam_num = int(match.group(1))
                    existing_dirs.add(cam_num)
                    print(f"Found existing camera directory: {dirname} (camera {cam_num})")
        except Exception as e:
            print(f"Error while finding existing camera directories: {e}")
            
        return existing_dirs
        
    def _get_next_available_camera_number(self, preferred_num, existing_numbers):
        """Get the next available camera number, starting with the preferred number."""
        if preferred_num not in existing_numbers:
            return preferred_num
            
        # If preferred number is taken, find the next available number
        next_num = max(existing_numbers) + 1 if existing_numbers else 1
        print(f"Camera {preferred_num} directory already exists, using {next_num} instead")
        return next_num

    def _load_subset_files(self):
        """Load subset files from the z_folder"""
        if self.z_folder:
            try:
                import glob
                import re
                subset_pattern = os.path.join(self.z_folder, 'subset_*.json')
                subset_files = glob.glob(subset_pattern)
                
                # Sort numerically instead of alphabetically
                def extract_number(filename):
                    match = re.search(r'subset_(\d+)\.json', os.path.basename(filename))
                    return int(match.group(1)) if match else 0
                
                self.subset_files = sorted(subset_files, key=extract_number)
                print(f"Loaded {len(self.subset_files)} subset files from {self.z_folder}")
                if self.subset_files:
                    print(f"First few files: {[os.path.basename(f) for f in self.subset_files[:5]]}")
            except Exception as e:
                print(f"Error loading subset files: {e}")
        else:
            print("No z_folder specified, skipping subset file loading")

    def _get_subset_file(self, index):
        """Get a subset file based on the current subset index"""
        if self.subset_files:
            return self.subset_files[index % len(self.subset_files)]
        else:
            print("No subset files available")
            return None

    def _get_subset_config(self, index):
        """Get a subset configuration based on the current subset index"""
        file_path = self._get_subset_file(index)
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading subset configuration: {e}")
        else:
            print("No subset configuration available")
            return None

    def _set_subset_config(self, subset_config):
        """Set the current subset configuration"""
        self.current_subset_config = subset_config
        print(f"Set current subset config with {len(subset_config)} cameras")

    def _get_current_subset_config(self):
        """Get the current subset configuration"""
        return self.current_subset_config

    def _get_current_subset_index(self):
        """Get the current subset index"""
        return self.current_subset_index

    def _increment_subset_index(self):
        """Increment the current subset index"""
        self.current_subset_index = (self.current_subset_index + 1) % len(self.subset_files)
        
    def _parse_camera_config_from_subset(self, subset_config):
        """Parse camera configuration from subset format to usable format
        
        Input format: [camera_id, fov, width, height, exposure_level, focal_length, loc1, loc2, loc3, loc4, detection_output]
        Returns dictionary with camera configurations mapped to placement positions
        """
        camera_configs = {}
        
        for i, cam_config in enumerate(subset_config):
            camera_id, fov, width, height, exposure, focal_length, loc1, loc2, loc3, loc4, detection_output = cam_config
            
            # Determine which placement (1-4) based on loc1-loc4 one-hot encoding
            placement_num = None
            loc_values = [loc1, loc2, loc3, loc4]
            for j, val in enumerate(loc_values):
                if val == 1:
                    placement_num = j + 1  # placement1, placement2, etc.
                    break
                    
            if placement_num is None:
                print(f"Warning: Camera {camera_id} has no valid placement location")
                continue
                
            # Get placement position and rotation from the config file
            placement_key = f"placement{placement_num}"
            if placement_key not in self.camera_config:
                print(f"Warning: {placement_key} not found in camera config")
                continue
                
            placement_info = self.camera_config[placement_key]
            
            camera_key = f"camera{camera_id}"
            camera_configs[camera_key] = {
                "camera_id": camera_id,
                "fov": fov,
                "resolution": {"width": width, "height": height},
                "exposure": exposure,
                "focal_length": focal_length,
                "placement": placement_info,
                "placement_number": placement_num
            }
            
        return camera_configs
        
    def _configure_camera_with_unrealcv(self, camera_id, config):
        """Configure a camera using UnrealCV commands based on subset configuration"""
        try:
            print(f"Configuring camera {camera_id} with UnrealCV...")
            
            # First, spawn cameras to ensure they exist
            print(f"Spawning cameras...")
            self.unrealcv_client.request('vset /cameras/spawn')
            self.unrealcv_client.request('vset /cameras/spawn')
            
            # Get list of available cameras for debugging
            camera_list = self.unrealcv_client.request('vget /cameras')
            print(f"Available cameras after spawn: {camera_list}")
            
            # Set FOV
            fov_cmd = f'vset /camera/{camera_id}/fov {config["fov"]}'
            fov_result = self.unrealcv_client.request(fov_cmd)
            print(f"FOV command: {fov_cmd} -> {fov_result}")
            
            # Set Resolution
            width = config["resolution"]["width"]
            height = config["resolution"]["height"]
            size_cmd = f'vset /camera/{camera_id}/size {width} {height}'
            size_result = self.unrealcv_client.request(size_cmd)
            print(f"Size command: {size_cmd} -> {size_result}")
            
            # Set Exposure
            exposure_cmd = f'vset /camera/{camera_id}/exposure_bias {config["exposure"]}'
            exposure_result = self.unrealcv_client.request(exposure_cmd)
            print(f"Exposure command: {exposure_cmd} -> {exposure_result}")
            
            # Set Focal Length (using focal distance with default range of 200.0)
            focal_range = 200.0
            focal_cmd = f'vset /camera/{camera_id}/focal {config["focal_length"]} {focal_range}'
            focal_result = self.unrealcv_client.request(focal_cmd)
            print(f"Focal command: {focal_cmd} -> {focal_result}")
            
            # Set Camera Position and Rotation
            placement = config["placement"]
            location = placement["location"]
            rotation = placement["rotation"]
            
            # Set position
            pos_cmd = f'vset /camera/{camera_id}/location {location["x"]} {location["y"]} {location["z"]}'
            pos_result = self.unrealcv_client.request(pos_cmd)
            print(f"Position command: {pos_cmd} -> {pos_result}")
            
            # Set rotation
            rot_cmd = f'vset /camera/{camera_id}/rotation {rotation["pitch"]} {rotation["yaw"]} {rotation["roll"]}'
            rot_result = self.unrealcv_client.request(rot_cmd)
            print(f"Rotation command: {rot_cmd} -> {rot_result}")
            
            print(f"Successfully configured camera {camera_id}")
            
        except Exception as e:
            print(f"Error configuring camera {camera_id}: {e}")
            
    def _setup_cameras_from_subset(self, subset_config):
        """Setup all cameras based on subset configuration"""
        print("Setting up cameras from subset configuration...")
        
        # Parse the subset configuration
        camera_configs = self._parse_camera_config_from_subset(subset_config)
        
        print(f"Found {len(camera_configs)} cameras to configure:")
        for cam_name, config in camera_configs.items():
            print(f"  {cam_name}: placement {config['placement_number']}, FOV {config['fov']}, " +
                  f"resolution {config['resolution']['width']}x{config['resolution']['height']}")
        
        # Configure each camera
        for cam_name, config in camera_configs.items():
            self._configure_camera_with_unrealcv(config["camera_id"], config)
            
        return camera_configs

    def _get_camera_capture_dirs(self, camera_id):
        """Get the appropriate directories for capturing images"""
        print(f"Looking for camera {camera_id} in camera_dirs...")
        print(f"Available cameras in camera_dirs: {list(self.camera_dirs.keys())}")
        
        if camera_id in self.camera_dirs:
            camera_dir = self.camera_dirs[camera_id]
            print(f"Found {camera_id} in camera_dirs")
            return {
                'rgb': camera_dir['rgb'],
                'mask': camera_dir['mask'],
                'state': camera_dir.get('state')
            }
        else:
            print(f"ERROR: Camera {camera_id} not found in camera_dirs")
            print(f"camera_dirs contents: {self.camera_dirs}")
            return {'rgb': '', 'mask': '', 'state': ''}
            
    def _setup_subset_directories(self, subset_filename, active_cameras):
        """Create subset-specific directories for active cameras
        
        New structure: base_dir/subset_#/camera_#/{rgb,mask,state,camera_config.json}
        """
        subset_name = os.path.splitext(os.path.basename(subset_filename))[0]  # e.g., "subset_1"
        print(f"Setting up directories for {subset_name}...")
        
        # Create the main subset directory
        subset_dir = os.path.join(self.base_output_dir, subset_name)
        os.makedirs(subset_dir, exist_ok=True)
        print(f"Created subset directory: {subset_dir}")
        
        # Verify subset directory exists
        if not os.path.exists(subset_dir):
            print(f"ERROR: Failed to create subset directory: {subset_dir}")
            return
        
        # Clear any existing camera_dirs for this subset
        print(f"Clearing previous camera_dirs. Previous count: {len(self.camera_dirs)}")
        self.camera_dirs = {}
        
        # Create camera directories within the subset
        for cam_name, config in active_cameras.items():
            camera_id = config['camera_id']
            camera_dir_name = f"camera_{camera_id}"
            
            print(f"  Processing {cam_name} with camera_id {camera_id}...")
            
            # Create camera-specific directory structure
            camera_base_dir = os.path.join(subset_dir, camera_dir_name)
            rgb_dir = os.path.join(camera_base_dir, 'rgb')
            mask_dir = os.path.join(camera_base_dir, 'mask')
            state_dir = os.path.join(camera_base_dir, 'state') if self.save_state else None
            
            # Create all directories
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            if self.save_state:
                os.makedirs(state_dir, exist_ok=True)
            
            # Verify directories were created
            if not os.path.exists(rgb_dir):
                print(f"    ERROR: Failed to create RGB directory: {rgb_dir}")
            if not os.path.exists(mask_dir):
                print(f"    ERROR: Failed to create mask directory: {mask_dir}")
            if self.save_state and state_dir and not os.path.exists(state_dir):
                print(f"    ERROR: Failed to create state directory: {state_dir}")
            
            # Store directory structure in camera_dirs using camera1, camera2 etc. format
            camera_key = f"camera{camera_id}"
            self.camera_dirs[camera_key] = {
                'camera_dir': camera_base_dir,
                'rgb': rgb_dir,
                'mask': mask_dir,
                'state': state_dir,
                'camera_config_file': os.path.join(camera_base_dir, 'camera_config.json')
            }
            
            print(f"  Created directories for {camera_key}:")
            print(f"    Base: {camera_base_dir}")
            print(f"    RGB: {rgb_dir}")
            print(f"    Mask: {mask_dir}")
            if self.save_state:
                print(f"    State: {state_dir}")
                
            # Create camera_config.json file in the camera directory
            self._create_camera_config_json(camera_key, camera_base_dir)
        
        # Show final camera_dirs structure
        print(f"Final camera_dirs structure:")
        for cam_key, dirs in self.camera_dirs.items():
            print(f"  {cam_key}: RGB={dirs['rgb']}, Mask={dirs['mask']}")
            
        print(f"Total cameras set up: {len(self.camera_dirs)}")

    def _load_all_subset_cameras(self):
        """Load and setup all cameras from all subset files for launch-all-parallel mode"""
        print(f"\n=== Loading all cameras from {len(self.subset_files)} subset files ===")
        
        camera_counter = 0  # Global camera counter to avoid ID conflicts
        
        for subset_index, subset_file in enumerate(self.subset_files):
            subset_name = os.path.splitext(os.path.basename(subset_file))[0]
            print(f"Processing {subset_name} ({subset_index + 1}/{len(self.subset_files)})...")
            
            try:
                # Load subset configuration
                with open(subset_file, 'r') as f:
                    subset_config = json.load(f)
                
                # Parse cameras from this subset
                subset_cameras = self._parse_camera_config_from_subset(subset_config)
                
                # Create directories for this subset
                subset_dirs = {}
                for cam_name, config in subset_cameras.items():
                    original_camera_id = config['camera_id']
                    
                    # Assign unique global camera ID for UnrealCV
                    global_camera_id = camera_counter
                    camera_counter += 1
                    
                    # Create directory structure: subset_#/camera_#/{rgb,mask,state}
                    subset_dir = os.path.join(self.base_output_dir, subset_name)
                    camera_dir_name = f"camera_{original_camera_id}"
                    camera_base_dir = os.path.join(subset_dir, camera_dir_name)
                    
                    rgb_dir = os.path.join(camera_base_dir, 'rgb')
                    mask_dir = os.path.join(camera_base_dir, 'mask')
                    state_dir = os.path.join(camera_base_dir, 'state') if self.save_state else None
                    
                    # Create directories
                    os.makedirs(rgb_dir, exist_ok=True)
                    os.makedirs(mask_dir, exist_ok=True)
                    if self.save_state:
                        os.makedirs(state_dir, exist_ok=True)
                    
                    # Store camera info with global ID mapping
                    camera_info = {
                        'subset_name': subset_name,
                        'original_camera_id': original_camera_id,
                        'global_camera_id': global_camera_id,
                        'config': config,
                        'rgb_dir': rgb_dir,
                        'mask_dir': mask_dir,
                        'state_dir': state_dir,
                        'camera_base_dir': camera_base_dir
                    }
                    
                    # Use a unique key combining subset and camera ID
                    camera_key = f"{subset_name}_camera{original_camera_id}"
                    self.all_subset_cameras[camera_key] = camera_info
                    
                    print(f"  {cam_name} → global_id={global_camera_id}, dirs={camera_base_dir}")
                    
                    # Create camera_config.json for this camera
                    self._create_camera_config_json_for_subset(camera_info)
                    
            except Exception as e:
                print(f"Error processing {subset_file}: {e}")
                continue
        
        print(f"=== Loaded {len(self.all_subset_cameras)} total cameras from all subsets ===")
        return self.all_subset_cameras
        
    def _create_camera_config_json_for_subset(self, camera_info):
        """Create camera_config.json for a specific subset camera"""
        try:
            config = camera_info['config']
            
            camera_config_data = {
                "resolution": config["resolution"],
                "fov_deg": config["fov"],
                "position": config["placement"]["location"],
                "orientation": config["placement"]["rotation"],
                "exposure": config["exposure"],
                "focal_length": config["focal_length"],
                "placement_number": config["placement_number"],
                "subset_name": camera_info['subset_name'],
                "original_camera_id": camera_info['original_camera_id'],
                "global_camera_id": camera_info['global_camera_id']
            }
            
            # Save to JSON file
            camera_config_path = os.path.join(camera_info['camera_base_dir'], "camera_config.json")
            with open(camera_config_path, 'w') as f:
                json.dump(camera_config_data, f, indent=2)
                
        except Exception as e:
            print(f"Error creating camera_config.json for {camera_info['subset_name']}: {e}")
            
    def _setup_all_subset_cameras(self):
        """Setup all cameras from all subsets in UnrealCV"""
        print(f"\n=== Setting up {len(self.all_subset_cameras)} cameras in UnrealCV ===")
        
        # First spawn enough cameras
        print("Spawning cameras...")
        for i in range(len(self.all_subset_cameras)):
            spawn_result = self.unrealcv_client.request('vset /cameras/spawn')
            if i % 10 == 0:  # Progress indicator
                print(f"  Spawned {i+1}/{len(self.all_subset_cameras)} cameras...")
        
        # Get list of available cameras for debugging
        camera_list = self.unrealcv_client.request('vget /cameras')
        print(f"Available cameras after spawn: {camera_list}")
        
        # Configure each camera
        for camera_key, camera_info in self.all_subset_cameras.items():
            try:
                global_id = camera_info['global_camera_id']
                config = camera_info['config']
                
                print(f"Configuring {camera_key} (global_id={global_id})...")
                
                # Set FOV
                self.unrealcv_client.request(f'vset /camera/{global_id}/fov {config["fov"]}')
                
                # Set Resolution
                width = config["resolution"]["width"]
                height = config["resolution"]["height"]
                self.unrealcv_client.request(f'vset /camera/{global_id}/size {width} {height}')
                
                # Set Exposure
                self.unrealcv_client.request(f'vset /camera/{global_id}/exposure_bias {config["exposure"]}')
                
                # Set Focal Length
                focal_range = 200.0
                self.unrealcv_client.request(f'vset /camera/{global_id}/focal {config["focal_length"]} {focal_range}')
                
                # Set Position and Rotation
                placement = config["placement"]
                location = placement["location"]
                rotation = placement["rotation"]
                
                self.unrealcv_client.request(f'vset /camera/{global_id}/location {location["x"]} {location["y"]} {location["z"]}')
                self.unrealcv_client.request(f'vset /camera/{global_id}/rotation {rotation["pitch"]} {rotation["yaw"]} {rotation["roll"]}')
                
            except Exception as e:
                print(f"Error configuring {camera_key}: {e}")
        
        print("=== All cameras configured ===\n")

    def _execute_mission_launch_all_parallel(self):
        """Execute mission capturing from ALL subset cameras at each waypoint"""
        print("\n=== Starting launch-all-parallel mission execution ===")
        
        # We need AirSim client to move the drone
        if not self.airsim_client:
            print("ERROR: No AirSim client available. Cannot execute drone mission.")
            return
            
        mission_file = self.config['drone_config']['mission_file']
        
        # Load mission data
        try:
            if not os.path.exists(mission_file):
                print(f"WARNING: Mission file not found at {mission_file}")
                alternative_path = os.path.join(os.path.dirname(__file__), mission_file)
                if os.path.exists(alternative_path):
                    mission_file = alternative_path
                else:
                    print(f"ERROR: Could not find mission file at any location")
                    return
        except Exception as e:
            print(f"Error checking mission file: {e}")
            return
            
        with open(mission_file, 'r') as f:
            mission_data = json.load(f)
        
        # Handle mission wrapper
        if "mission" in mission_data:
            mission_data = mission_data["mission"]
        
        target_drone = mission_data['drones'][0]
        waypoints = target_drone['waypoints']
        print(f"Found {len(waypoints)} waypoints for mission")
        
        max_images = self.config['data_collection'].get('max_images', float('inf'))
        start_time = time.time()
        
        # Track frame indices for each camera
        frame_indices = {}
        for camera_key in self.all_subset_cameras.keys():
            frame_indices[camera_key] = 0
        
        # Navigate through waypoints
        for i, waypoint in enumerate(waypoints):
            # Check if max images or duration reached
            if max(frame_indices.values()) >= max_images or (time.time() - start_time) >= self.capture_duration:
                print("Reached max images or duration limit. Ending mission.")
                break
                
            print(f"\n>>> WAYPOINT {waypoint['number']} ({i+1}/{len(waypoints)}) - LAUNCH ALL PARALLEL <<<")
            
            velocity = waypoint.get('speed', 5)
            
            # Extract waypoint coordinates
            x = waypoint.get('x', 0)
            y = waypoint.get('y', 0)
            altitude = waypoint.get('altitude', 0)
            
            if 'position' in waypoint:
                if 'x' in waypoint['position']: x = waypoint['position']['x']
                if 'y' in waypoint['position']: y = waypoint['position']['y']
                if 'z' in waypoint['position']: altitude = waypoint['position']['z']
            
            print(f"Moving to coordinates: X={x}, Y={y}, Alt={altitude}")
            
            # Move drone to waypoint
            self._move_to_waypoint(x=x, y=y, altitude=altitude, velocity=velocity)
            
            print(f"Arrived at waypoint {waypoint['number']}. Waiting 3 seconds...")
            time.sleep(3)
            
            # Pause simulation for capture
            print(f"Pausing simulation for mass capture from {len(self.all_subset_cameras)} cameras...")
            self.unrealcv_client.request('vset /action/game/pause')
            time.sleep(0.5)
            
            # Capture from ALL cameras
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"Starting capture from all {len(self.all_subset_cameras)} cameras...")
            capture_count = 0
            
            for camera_key, camera_info in self.all_subset_cameras.items():
                try:
                    global_id = camera_info['global_camera_id']
                    frame_idx = frame_indices[camera_key]
                    
                    # Skip if this camera has reached max images
                    if frame_idx >= max_images:
                        continue
                    
                    # Create base filename
                    base_filename = f"{frame_idx}_{timestamp}_camera{camera_info['original_camera_id']}"
                    
                    # Capture Mask
                    mask_filename = f"{base_filename}_object_mask.png"
                    mask_path = os.path.join(camera_info['mask_dir'], mask_filename)
                    mask_result = self.unrealcv_client.request(f'vget /camera/{global_id}/object_mask {mask_path}')
                    
                    # Capture RGB
                    rgb_filename = f"{base_filename}_lit.png"
                    rgb_path = os.path.join(camera_info['rgb_dir'], rgb_filename)
                    rgb_result = self.unrealcv_client.request(f'vget /camera/{global_id}/lit {rgb_path}')
                    
                    # Save State if enabled
                    if self.save_state and camera_info['state_dir']:
                        state_filename = f"{base_filename}_state.json"
                        state_path = os.path.join(camera_info['state_dir'], state_filename)
                        self._save_state_to_json_for_camera(camera_info, state_path, timestamp, frame_idx)
                    
                    # Increment frame index for this camera
                    frame_indices[camera_key] += 1
                    capture_count += 1
                    
                    if capture_count % 20 == 0:  # Progress indicator
                        print(f"  Captured from {capture_count}/{len(self.all_subset_cameras)} cameras...")
                    
                except Exception as e:
                    print(f"Error capturing from {camera_key}: {e}")
            
            print(f"Completed capture from {capture_count} cameras")
            
            # Resume simulation
            print(f"Resuming simulation...")
            self.unrealcv_client.request('vset /action/game/resume')
            time.sleep(0.5)
            
            print(f"Completed waypoint {waypoint['number']}")
        
        print(f"\n=== Completed launch-all-parallel mission ===")
        
        # Print summary statistics
        print("Capture summary by subset:")
        subset_stats = {}
        for camera_key, frame_count in frame_indices.items():
            subset_name = camera_key.split('_camera')[0]
            if subset_name not in subset_stats:
                subset_stats[subset_name] = []
            subset_stats[subset_name].append(frame_count)
        
        for subset_name, counts in subset_stats.items():
            print(f"  {subset_name}: {len(counts)} cameras, {min(counts)}-{max(counts)} frames each")
            
    def _save_state_to_json_for_camera(self, camera_info, state_path, timestamp, frame_index):
        """Save state data for a specific camera in launch-all-parallel mode"""
        if not self.save_state:
            return
            
        try:
            global_id = camera_info['global_camera_id']
            
            # Get camera position and rotation
            pos_result = self.unrealcv_client.request(f'vget /camera/{global_id}/location')
            rot_result = self.unrealcv_client.request(f'vget /camera/{global_id}/rotation')
            fov_result = self.unrealcv_client.request(f'vget /camera/{global_id}/fov')
            
            # Parse position and rotation
            pos_parts = pos_result.split(' ') if pos_result != 'error' else ['0', '0', '0']
            rot_parts = rot_result.split(' ') if rot_result != 'error' else ['0', '0', '0']
            
            camera_position = {
                'x': float(pos_parts[0]) if len(pos_parts) >= 3 else 0.0,
                'y': float(pos_parts[1]) if len(pos_parts) >= 3 else 0.0,
                'z': float(pos_parts[2]) if len(pos_parts) >= 3 else 0.0
            }
            
            camera_rotation = {
                'pitch': float(rot_parts[0]) if len(rot_parts) >= 3 else 0.0,
                'yaw': float(rot_parts[1]) if len(rot_parts) >= 3 else 0.0,
                'roll': float(rot_parts[2]) if len(rot_parts) >= 3 else 0.0
            }
            
            camera_fov = float(fov_result) if fov_result != 'error' else 90.0
            
            # Get drone state
            drone_state = self._get_drone_state()
            
            # Create state data
            state_data = {
                'timestamp': timestamp,
                'frame_index': frame_index,
                'subset_info': {
                    'subset_name': camera_info['subset_name'],
                    'original_camera_id': camera_info['original_camera_id'],
                    'global_camera_id': camera_info['global_camera_id']
                },
                'camera': {
                    'id': f"camera{camera_info['original_camera_id']}",
                    'position': camera_position,
                    'rotation': camera_rotation,
                    'fov': camera_fov
                },
                'drone': drone_state
            }
            
            # Save to JSON file
            with open(state_path, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving state for {camera_info['subset_name']}_camera{camera_info['original_camera_id']}: {e}")

if __name__ == "__main__":
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run data collection from Unreal Engine')
    parser.add_argument('--config', type=str, default="scripts\Data_collection\data_collection_config\config8-brushify-lake.json",
                        help='Path to the configuration file')
    parser.add_argument('--state', action='store_true', 
                        help='Enable state saving to JSON files')
    parser.add_argument('--visualize-line', action='store_true',
                        help='Draw arrows from drone to destination points')
    parser.add_argument('--parallel', action='store_true',
                        help='Run in parallel mode - all cameras capture at each waypoint')
    parser.add_argument('--launch-all-parallel', action='store_true',
                        help='Launch all cameras from all subsets and capture from all simultaneously (requires --z-folder)')
    parser.add_argument('--z-folder', type=str, default=None,
                        help='Path to the z_folder containing subset configuration files')
    args = parser.parse_args()

    start_time = time.time()
    try:
        # Initialize data collection with appropriate flags
        data_collection = FullDataCollection(args.config, save_state=args.state, 
                                            visualize_line=args.visualize_line,
                                            parallel_mode=args.parallel,
                                            z_folder=args.z_folder,
                                            launch_all_parallel=args.launch_all_parallel)
        
        # Print mode message
        if args.state:
            print("Running with state saving ENABLED (saving AirSim state to JSON)")
        else:
            print("Running with state saving DISABLED (not saving state to JSON, but drone will still move)")
            
        if args.visualize_line:
            print("Visualization ENABLED (drawing arrows from drone to destination)")
            
        if args.parallel:
            print("PARALLEL MODE ENABLED (all cameras will capture at each waypoint)")
        else:
            print("SEQUENTIAL MODE ENABLED (completing full mission for each camera separately)")
            
        if args.launch_all_parallel:
            print("LAUNCH-ALL-PARALLEL MODE ENABLED (all cameras from all subsets will capture simultaneously)")
            if not args.z_folder:
                print("WARNING: --launch-all-parallel requires --z-folder to be specified!")
                
        data_collection.run()
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal script execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}") 