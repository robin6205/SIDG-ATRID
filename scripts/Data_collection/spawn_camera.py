###################################################
# Spawn_camera.py
# Developer: Joshua Chang
# Date: 4/21/2024
# Description: This script will spawn a camera at specified locations and capture images
# On 'Purdue_airport3_7'
##################################################
from __future__ import division, absolute_import, print_function
from src.utils import lat_long_to_local_xy_rotated, create_vehicle, read_coordinates_from_json
import src.airsim as airsim
import math

framework = 'GT'  # 'GT' or 'CV

agent_config_path = 'test_coordinates/uav_path_test_4.json'
simulation_config_path = 'config/sim_config.json'

sim_config = read_coordinates_from_json(simulation_config_path, 'sim_config')
print(sim_config)
# Reference origin in Unreal Engine (location of Georefernce)
origin_lat = sim_config["origin_latitude"]
origin_long = sim_config["origin_longitude"]
# Assume Unreal units match meters (scale=1)
scale = sim_config["scale"]


# Sensor placement
sensor_location = read_coordinates_from_json(agent_config_path, 'sensors')
for sensor in sensor_location:
    target_lat = sensor["pose"]["x"]
    target_long = sensor["pose"]["y"]
    altitude = sensor["pose"]["z"]
    
    roll = math.radians(sensor["orientation"]["roll"])
    pitch = math.radians(sensor["orientation"]["pitch"])
    yaw = math.radians(sensor["orientation"]["yaw"])
    
    # Convert geographic coordinates to local X, Y coordinates
    x_rot, y_rot = lat_long_to_local_xy_rotated(target_lat, target_long, origin_lat, origin_long, scale)
    z_offset = -65  # Offset where the playerstart is located
    z_adjusted = altitude + z_offset  # Adjusted z coordinate for Unreal units

    # Coordinates and orientation for where you want to spawn the object
    x, y, z = x_rot / 100, y_rot / 100, z_adjusted / 100    
    
    
    # spawn camera using UnrealCV or GT
    if framework == 'GT':
        # Asset name of the blueprint object in the Unreal project database
        # This should be the name of the asset as seen in Unreal's content browser
        asset_name = "Recording_camera5"

        # Desired name for the new object instance
        object_name = "RecordingCamera5Instance"

        # Connect to the AirSim client
        client = airsim.VehicleClient()
        client.confirmConnection()
        
        # Spawn the blueprint object at the specified pose and scale
        spawned_object_name = client.simSpawnObject(object_name=object_name, 
                                                    asset_name=asset_name, 
                                                    pose=airsim.Pose(airsim.Vector3r(x, y, -z), airsim.to_quaternion(pitch,roll, yaw)), 
                                                    scale=airsim.Vector3r(1, 1, 1),
                                                    is_blueprint=True)  
        if spawned_object_name:
            print(f"Successfully spawned '{spawned_object_name}' at location ({x}, {y}, {z}) oriented at ({roll}, {pitch}, {yaw})")
        else:
            print("Failed to spawn the object. Please check the asset name and parameters.")
            break

    elif framework == 'CV':
        
        from unrealcv import Client
        from PIL import Image
        import io
        import matplotlib.pyplot as plt
        import sys
        import os
        import time
        import numpy as np
        from io import BytesIO, StringIO
        from datetime import datetime


        def imread8(im_file):
            ''' Read image as an 8-bit numpy array '''
            im = np.asarray(Image.open(im_file))
            return im

        def read_png(res):
            print('reading png')
            img = Image.open(io.BytesIO(res))
            return np.asarray(img)

        def read_npy(res):
            return np.load(io.BytesIO(res))

        def normalize_normal_map(normal_map):
            # Normalize the normal map values to the range [0, 1]
            norm_map = (normal_map / 255.0) * 2 - 1  # Map to [-1, 1]
            return (norm_map + 1) / 2  # Map to [0, 1] for display

        def save_image(image_data, folder='data'):
            if not os.path.exists(folder):
                os.makedirs(folder)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(folder, f'rgb_{timestamp}.png')
            img = Image.open(io.BytesIO(image_data))
            img.save(filename)
            print(f'Saved {filename}')
            
        def normalize_depth(depth, clip_min=0, clip_max=500):
            # Clip depth values to avoid extreme outliers
            depth = np.clip(depth, clip_min, clip_max)
            depth_min = np.min(depth)
            depth_max = np.max(depth)
            return (depth - depth_min) / (depth_max - depth_min)

        # Initialize client
        client = Client(('localhost', 9000))

        try:
            client.connect()
            
            camera_id = '1'

            # Check if connected
            if not client.isconnected():
                print('UnrealCV server is not running.')
                sys.exit(-1)
            # Spawn a new camera
            client.request('vset /objects/spawn FusionCameraActor Cam1')
            # The actual id counts up from 1
            time.sleep(1)  # Give some time for the camera to be spawned
            # Set camera location and orientation
            location_command = f'vset /camera/{camera_id}/location {x_rot} {y_rot} {altitude}'
            print(location_command)
            rotation_command = f'vset /camera/{camera_id}/rotation {math.degrees(0)} {math.degrees(30)} {math.degrees(0)}'
            print(rotation_command)
            client.request(location_command)
            client.request(rotation_command)

            
            # get object names
            res = client.request('vget /objects')
            print(res)
            
            # use airsim to list all objects
            # client = airsim.VehicleClient()
            # client.confirmConnection()
            # print(client.simListSceneObjects())
            # disconnect

            
            # Get status
            res = client.request('vget /unrealcv/status')
            print(res)
            
            # get uclass name vget /object/[obj_name]/uclass_name
            res = client.request(f'vget /object/{camera_id}/uclass_name')
            print(res)
            
            # Get image
            res = client.request(f'vget /camera/{camera_id}/lit png')
            im = read_png(res)
            print('RGB image shape:', im.shape)
            
            # show image
            plt.imshow(im)
            plt.axis('off')
            plt.show()
            
            # Get image
            
            # res = client.request('vget /camera/1/object_mask png')
            # object_mask = read_png(res)
            

            # # Get depth
            # res = client.request('vget /camera/1/depth npy')
            # depth = read_npy(res)
            # print(depth)

            # # Normalize depth values for visualization
            # normalized_depth = normalize_depth(depth, clip_min=0, clip_max=2000)

            # Visualize the depth map
            # plt.imshow(normalized_depth, cmap='viridis')
            # plt.colorbar(label='Normalized Depth')
            # plt.axis('off')
            # plt.title('Depth Map')
            # plt.show()
            
            
            # Capture images for 30 seconds at 30 FPS
            fps = 30
            duration = 60
            start_time = time.time()

            while time.time() - start_time < duration:
                res = client.request('vget /camera/1/lit png')
                res_norm = client.request('vget /camera/1/object_mask png')
                
                if isinstance(res, str):
                    print('Received string response instead of bytes:', res)
                else:
                    
                    # save_image(res, folder='data/rgb')
                    save_image(res_norm, folder='data/normal')
                
                time.sleep(1 / fps)

        except KeyboardInterrupt:
            print("Interrupted by user")

        except Exception as e:
            print(e)

        finally:
            # Ensure proper cleanup and termination
            client.disconnect()
            print("Client disconnected")
            sys.exit(0)


# spawn uav vehicle at specified location
# vehicle_location = read_coordinates_from_json(agent_config_path, 'vehicle_spawn')
# vehicle_location_lat = vehicle_location[0]["x"]
# vehicle_location_long = vehicle_location[0]["y"]
# vehicle_yaw = math.radians(0)
# vehicle_x, vehicle_y = lat_long_to_local_xy_rotated(vehicle_location_lat, vehicle_location_long, origin_lat, origin_long, scale)
# vehicle_location = airsim.Vector3r(vehicle_x / 100, vehicle_y / 100, -z_offset / 100)

# create_vehicle(client=client, vehicle_name='drone2', location=vehicle_location, rotation=airsim.to_quaternion(0, 0, vehicle_yaw))
