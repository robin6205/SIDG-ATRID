from __future__ import division, absolute_import, print_function
import os
import sys
import time
import re
import json
import numpy as np
from PIL import Image
from datetime import datetime
import signal

# Define the base data output directory
BASE_OUTPUT_DIR = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput"
AGENT_LIST_PATH = r"D:\Unreal Projects\ACREDataCollection\AgentList.json"

class UnrealCVDataCollector:
    def __init__(self, client, base_output_dir, agent_list_path, weather_condition="clearsky"):
        self.client = client
        self.base_output_dir = base_output_dir
        self.agent_list_path = agent_list_path
        self.weather_condition = weather_condition
        self.weather_output_dir = os.path.join(self.base_output_dir, self.weather_condition)
        self.rgb_output_dir = os.path.join(self.weather_output_dir, 'rgb')
        self.mask_output_dir = os.path.join(self.weather_output_dir, 'mask')
        self.agent_color_info_path = os.path.join(self.weather_output_dir, "agent_color_info.json")
        self.frame_rate = 10
        self.capture_duration = 200

        # Create directories
        os.makedirs(self.rgb_output_dir, exist_ok=True)
        os.makedirs(self.mask_output_dir, exist_ok=True)

    def load_agent_list(self):
        """Load the agent list from JSON."""
        with open(self.agent_list_path, 'r') as agent_file:
            return json.load(agent_file)

    def get_agent_colors(self, agent_list):
        """Retrieve colors of all agents and save them to a JSON file."""
        agent_color_data = {}
        for agent_id, agent_name in agent_list.items():
            try:
                color_str = self.client.request(f'vget /object/{agent_id}/color')
                match = re.match(r'\(R=(\d+),G=(\d+),B=(\d+),A=(\d+)\)', color_str)
                if match:
                    color_data = {
                        'R': int(match.group(1)),
                        'G': int(match.group(2)),
                        'B': int(match.group(3)),
                        'A': int(match.group(4))
                    }
                    agent_color_data[agent_name] = color_data
                    print(f"Agent {agent_name} (ID: {agent_id}) color: {color_data}")
            except Exception as e:
                print(f"Error retrieving color for agent {agent_name} (ID: {agent_id}): {e}")

        # Save agent colors to JSON
        with open(self.agent_color_info_path, 'w') as json_file:
            json.dump(agent_color_data, json_file, indent=4)
        print(f"Saved agent color data to {self.agent_color_info_path}")
        return agent_color_data

    def set_agent_color(self, agent_list, target_agent_name, target_color):
        """Set a specific color for a target agent."""
        for agent_id, agent_name in agent_list.items():
            if agent_name == target_agent_name:
                # Use the correct UnrealCV command for setting object color
                color_command = f'vset /object/{agent_id}/color {target_color[0]} {target_color[1]} {target_color[2]}'
                self.client.request(color_command)
                print(f"Set color for {agent_name} (ID: {agent_id}) to {target_color}")


    def set_camera_position(self, location, rotation):
        """Set the position and rotation of the camera."""
        self.client.request('vset /camera/0/location {x} {y} {z}'.format(**location))
        self.client.request('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**rotation))
        print(f"Camera set to location {location} and rotation {rotation}")

    def capture_frames(self):
        """Capture RGB and object mask frames for the specified duration."""
        start_time = time.time()
        frame_index = 0

        while (time.time() - start_time) < self.capture_duration:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.client.request('vrun pause')
            # Capture and save object mask image
            mask_filename = f"{frame_index}_{timestamp}_object_mask.png"
            mask_path = os.path.join(self.mask_output_dir, mask_filename)
            self.client.request(f'vget /camera/0/object_mask {mask_path}')

            # Capture and save RGB image
            rgb_filename = f"{frame_index}_{timestamp}_lit.png"
            rgb_path = os.path.join(self.rgb_output_dir, rgb_filename)
            self.client.request(f'vget /camera/0/lit {rgb_path}')

            frame_index += 1
            
            self.client.request('vrun pause')
            # time.sleep(1 / self.frame_rate)
        print(f"Captured frames saved to {self.rgb_output_dir} and {self.mask_output_dir}")


# Register the signal handler
def signal_handler(sig, frame):
    print('\nCleaning up and shutting down...')
    if client.isconnected():
        client.disconnect()
    print('Successfully disconnected from UnrealCV')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Main script
if __name__ == "__main__":
    try:
        from unrealcv import Client
        # client = Client(('192.168.0.84', 9000))  # Replace with your IP
        client = Client(('127.0.0.1', 9000))  # Replace with your IP
        client.connect()

        if not client.isconnected():
            print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
            sys.exit(-1)

        # Initialize the data collector
        data_collector = UnrealCVDataCollector(client, BASE_OUTPUT_DIR, AGENT_LIST_PATH, weather_condition="sunnycloudy2")

        # Load the agent list
        agent_list = data_collector.load_agent_list()


        # Set a specific color for the DJIOctocopter
        target_color = (255, 255, 0)  # Yellow
        data_collector.set_agent_color(agent_list, "DJIOctocopter", target_color)
        
        # Retrieve and confirm agent colors
        agent_colors = data_collector.get_agent_colors(agent_list)

        # Set the camera position and rotation
        camera_location = {'x': -2760.096251, 'y': -86226.228138, 'z': -3225.082072}
        camera_rotation = {'pitch': 40.956877, 'yaw': -42.862931, 'roll': 0.0}
        data_collector.set_camera_position(camera_location, camera_rotation)

        # Capture frames
        data_collector.capture_frames()

    except Exception as e:
        print(f'An error occurred: {e}')
    finally:
        if client.isconnected():
            client.disconnect()
            print('Disconnected from UnrealCV')
