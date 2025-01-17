from __future__ import division, absolute_import, print_function
import os
import sys
import time
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import signal
from datetime import datetime

# Define the base data output directory
base_output_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput"

# Helper functions
def imread8(im_file):
    '''Read image as an 8-bit numpy array'''
    im = np.asarray(Image.open(im_file))
    return im

def read_png(res):
    import io, PIL.Image
    img = PIL.Image.open(io.BytesIO(res))
    return np.asarray(img)

def read_image(file_path):
    return np.asarray(Image.open(file_path))

def signal_handler(sig, frame):
    print('\nCleaning up and shutting down...')
    if client.isconnected():
        client.disconnect()
    print('Successfully disconnected from UnrealCV')
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

try:
    from unrealcv import Client
    client = Client(('192.168.0.84', 9000))  # Replace with your IP
    client.connect()

    if not client.isconnected():
        print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
        sys.exit(-1)
    else:
        print('UnrealCV server is running')
        
    res = client.request('vget /unrealcv/status')
    print(res)
    # Parameters
    frame_rate = 10  # Frames per second
    capture_duration = 200  # Duration in seconds
    weather_condition = 'clearsky'  # Options: 'clearsky', 'rain', 'snow', 'fog', 'night'
    
    # Create weather-specific directories if they don't exist
    weather_output_dir = os.path.join(base_output_dir, weather_condition)
    rgb_output_dir = os.path.join(weather_output_dir, 'rgb')
    mask_output_dir = os.path.join(weather_output_dir, 'mask')

    os.makedirs(rgb_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Load agent list from JSON file
    agent_list_path = r"D:\Unreal Projects\ACREDataCollection\AgentList.json"
    with open(agent_list_path, 'r') as agent_file:
        agent_list = json.load(agent_file)
    # Get mask information and save object colors
    def get_agent_colors(agent_list, client):
        agent_color_data = {}

        for agent_id, agent_name in agent_list.items():
            try:
                color_str = client.request(f'vget /object/{agent_id}/color')
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

        return agent_color_data

    # Save agent color information to JSON
    agent_color_info = get_agent_colors(agent_list, client)
    color_json_path = os.path.join(weather_output_dir, "agent_color_info.json")
    with open(color_json_path, 'w') as json_file:
        json.dump(agent_color_info, json_file, indent=4)
    # print(f'Saved agent color data to {color_json_path}')


    # Set the location and rotation for the camera
    # loc = {'x': -1560.000000, 'y': -103720.000000, 'z': -2330.000000}
    # rot = {'pitch': 0.000000, 'yaw': -60.000000, 'roll': 0.000000}

    loc = {'x': -2760.096251, 'y': -86226.228138, 'z': -3225.082072}
    rot = {'pitch': 40.956877, 'yaw':-42.862931, 'roll': 0.0}

    # Set the position of the camera
    client.request('vset /camera/0/location {x} {y} {z}'.format(**loc))
    client.request('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**rot))

    start_time = time.time()
    frame_index = 0

    # # Capture frames at the specified frame rate for the given duration
    # while (time.time() - start_time) < capture_duration:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    #     # Capture and save RGB image
    #     rgb_filename = f"{frame_index}_{timestamp}_lit.png"
    #     rgb_path = os.path.join(rgb_output_dir, rgb_filename)
    #     client.request(f'vget /camera/0/lit {rgb_path}')
    #     # print(f'RGB image saved to {rgb_path}')
    #     # Capture and save object mask image
    #     mask_filename = f"{frame_index}_{timestamp}_object_mask.png"
    #     mask_path = os.path.join(mask_output_dir, mask_filename)
    #     client.request(f'vget /camera/0/object_mask {mask_path}')
    #     # print(f'Object mask image saved to {mask_path}')        

    #     frame_index += 1
    #     time.sleep(1 / frame_rate)  # Wait for the next frame

    try:
        # Capture frames at the specified frame rate for the given duration
        while (time.time() - start_time) < capture_duration:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            client.request('vrun pause')
            # client.request('vset /action/game/pause')
            
            # wait 0.5 seconds
            time.sleep(1)
            
            # Capture and save object mask image
            mask_filename = f"{frame_index}_{timestamp}_object_mask.png"
            mask_path = os.path.join(mask_output_dir, mask_filename)
            client.request(f'vget /camera/0/object_mask {mask_path}')
            
            
            # Capture and save RGB image
            rgb_filename = f"{frame_index}_{timestamp}_lit.png"
            rgb_path = os.path.join(rgb_output_dir, rgb_filename)
            client.request(f'vget /camera/0/lit {rgb_path}')
            # print(f'RGB image saved to {rgb_path}')
            
            # print(f'Object mask image saved to {mask_path}')        

            frame_index += 1
            client.request('vrun pause')
            # client.request('vset /action/game/pause')
    except KeyboardInterrupt:
        print('\nCapture interrupted by user. Exiting...')
        sys.exit(0)
        
        

except Exception as e:
    print(f'An error occurred: {e}')
finally:
    if 'client' in locals() and client.isconnected():
        client.disconnect()
        print('Disconnected from UnrealCV')
