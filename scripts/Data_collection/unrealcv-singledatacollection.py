from __future__ import division, absolute_import, print_function
import os, sys, time, re, json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import signal
from datetime import datetime

# Define the data output directory
output_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput"

# Helper functions
imread = plt.imread
def imread8(im_file):
    ''' Read image as a 8-bit numpy array '''
    im = np.asarray(Image.open(im_file))
    return im

def read_png(res):
    import io, PIL.Image
    img = PIL.Image.open(io.BytesIO(res))
    return np.asarray(img)

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

    if client.isconnected():
        # Set the location and rotation for the camera
        # client.request('vrun pause')
        # loc = {'x': -1900.000000, 'y': -110430.000000, 'z': -2740.000000}
        # rot = {'pitch': 0.000000, 'yaw': -60.000000, 'roll': 0.000000}
        loc = {'x': 1967.561189, 'y': -110719.557633, 'z': 1341.890300}
        rot = {'pitch': -11.444650, 'yaw': -46.403934, 'roll': 1.000000}
        
        # # Set the position of the camera
        client.request('vset /camera/0/location {x} {y} {z}'.format(**loc))
        client.request('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**rot))
        
        # Generate unique file index and timestamp
        index = 1  # Increment as needed
        date_time = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    
        # Capture and save RGB image
        rgb_filename = f"{index}_{date_time}_lit.png"
        rgb_path = os.path.join(output_dir, rgb_filename)
        client.request(f'vget /camera/0/lit {rgb_path}')
        # filename = rgb_path + ""
        # client.request(f'shot filename={rgb_path}')
        # print(f'RGB image saved to {rgb_path}')
        
        
        # Capture and save object mask image
        mask_filename = f"{index}_{date_time}_object_mask.png"
        mask_path = os.path.join(output_dir, mask_filename)
        mask = client.request(f'vget /camera/0/object_mask {mask_path}')
        print(f'Object mask image saved to {mask_path}')   
        

        # display mask
        mask = imread8(mask_path)
        plt.imshow(mask)
        plt.axis('off')
        plt.show()
        

        
        # Display the captured RGB image
        # res = client.request('vget /camera/0/lit png')
        # im = read_png(res)
        # plt.imshow(im)
        # plt.axis('off')
        # plt.show()
        
        # client.request('pause')
        


    # Postprocessing

    # import os
    # import re
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from PIL import Image
    # # Define the data output directory
    # output_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput"

    # # Helper function to read images
    # def read_image(file_path):
    #     return np.asarray(Image.open(file_path))

    # # Define the Color class
    # class Color:
    #     '''Utility class to parse color values'''
    #     regexp = re.compile(r'\(R=(.*),G=(.*),B=(.*),A=(.*)\)')

    #     def __init__(self, color_str):
    #         print(f"Received color string: {color_str}")  # Debug print
    #         match = self.regexp.match(color_str)
    #         if not match:
    #             raise ValueError(f"Invalid color format: {color_str}")
    #         self.R, self.G, self.B, self.A = [int(match.group(i)) for i in range(1, 5)]

    #     def __repr__(self):
    #         return f'({self.R}, {self.G}, {self.B}, {self.A})'

    # # Function to match colors in the object mask
    # def match_color(object_mask, target_color, tolerance=3):
    #     match_region = np.ones(object_mask.shape[:2], dtype=bool)
    #     for c in range(3):  # R, G, B channels
    #         min_val = target_color[c] - tolerance
    #         max_val = target_color[c] + tolerance
    #         channel_region = (object_mask[:, :, c] >= min_val) & (object_mask[:, :, c] <= max_val)
    #         match_region &= channel_region
    #     return match_region if match_region.sum() != 0 else None

    # # Process each image in the output directory
    # for filename in os.listdir(output_dir):
    #     if filename.endswith("_object_mask.png"):
    #         object_mask_path = os.path.join(output_dir, filename)
    #         object_mask = read_image(object_mask_path)

    #         # Retrieve object information from UnrealCV
    #         scene_objects = client.request('vget /objects').split(' ')
    #         id_to_color = {}
    #         for obj_id in scene_objects:
    #             try:
    #                 color_str = client.request(f'vget /object/{obj_id}/color')
    #                 color = Color(color_str)
    #                 id_to_color[obj_id] = color
    #             except Exception as e:
    #                 print(f"Error processing object {obj_id}: {e}")

    #         # Generate and visualize masks for each object
    #         id_to_mask = {}
    #         for obj_id, color in id_to_color.items():
    #             mask = match_color(object_mask, [color.R, color.G, color.B], tolerance=3)
    #             if mask is not None:
    #                 id_to_mask[obj_id] = mask
    #                 # plt.imshow(mask, cmap='gray')
    #                 # plt.title(f'Object ID: {obj_id}')
    #                 # plt.axis('off')
    #                 # plt.show()

    # # # Print all objects
    # scene_objects = client.request('vget /objects').split(' ')
    # print("All scene objects:")
    # for obj_id in scene_objects:
    #     try:
    #         # Get the name of the object
    #         obj_name = client.request(f'vget /object/{obj_id}/name')
    #         print(f"- {obj_id}: {obj_name}")
    #     except Exception as e:
    #         print(f"Error retrieving name for object {obj_id}: {e}")

    # # Then continue with the color processing
    # id_to_color = {}
    # for obj_id in scene_objects:
    #     try:
    #         color_str = client.request(f'vget /object/{obj_id}/color')
    #         print(f"Object {obj_id} color: {color_str}")  # Debug print
    #         color = Color(color_str)
    #         id_to_color[obj_id] = color
    #     except Exception as e:
    #         print(f"Error processing object {obj_id}: {e}")

    # else:
    #     print('Failed to connect to the UnrealCV server. Make sure it is running.')

except Exception as e:
    print(f'An error occurred: {e}')
finally:
    if 'client' in locals() and client.isconnected():
        client.disconnect()
        print('Disconnected from UnrealCV')
