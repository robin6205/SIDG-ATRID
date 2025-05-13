import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import math
import argparse # Added for command-line arguments
import os       # Added for os.path.exists

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Generate waypoints and visualize sensor FOV, optionally highlighting detected waypoints.")
parser.add_argument('--detection', action='store_true', help='Enable loading detection data and highlighting detected waypoints.')
args = parser.parse_args()

# --- Configuration ---
z = 260  # fixed Z-coordinate for grid
grid_spacing = 1800  # distance between grid points

# Camera Parameters
cameras = {
    "camera1": {
        "position": np.array([-74164.585036, -73303.671202, 773.166760]),
        "orientation": {'pitch': 7.0, 'yaw': 1.0, 'roll': 0.0},
        "resolution": {'width': 1920, 'height': 1080},
        "fov": 30.0
    },
    "camera2": {
        "position": np.array([-31023.204588, -30364.000000, 773.166760]),
        "orientation": {'pitch': 8.0, 'yaw': -90.0, 'roll': 0.0},
        "resolution": {'width': 1920, 'height': 1080},
        "fov": 30.0
    },
    "camera3": {
        "position": np.array([-31333.000000, -108495.000000, 773.166760]),
        "orientation": {'pitch': 28.0, 'yaw': 86.601774, 'roll': 0.0},
        "resolution": {'width': 1920, 'height': 1080},
        "fov": 90.0
    },
    "camera4": {
        "position": np.array([4244.677532, -73230.000000, 773.166760]),
        "orientation": {'pitch': 28.0, 'yaw': 177.0, 'roll': 0.0},
        "resolution": {'width': 1920, 'height': 1080},
        "fov": 90.0
    }
}

# Visual Parameters
schematic_fov_length = 5 * grid_spacing  # Visual length of the schematic FOV cone
extended_fov_line_length = 30 * grid_spacing  # Length for the extended dotted lines

# Define unique colors for each camera
camera_colors = {
    "camera1": {"cone": "cyan", "line": "navy"},
    "camera2": {"cone": "magenta", "line": "darkred"}, 
    "camera3": {"cone": "lime", "line": "green"},
    "camera4": {"cone": "orange", "line": "brown"}
}

# --- Helper Functions ---

def get_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    """Calculates the combined rotation matrix for Yaw, Pitch, Roll in degrees (UE convention: Z-up, left-handed)."""
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    roll = math.radians(roll_deg)

    # Rotation matrices (Left-Handed, Z-up)
    # Yaw around Z
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    R_yaw = np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ])

    # Pitch around Y
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    R_pitch = np.array([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ])

    # Roll around X
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    R_roll = np.array([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ])

    # Combine rotations (UE order: Yaw -> Pitch -> Roll)
    # To transform FROM camera local TO world, the combined matrix is R_yaw @ R_pitch @ R_roll
    R_combined = R_yaw @ R_pitch @ R_roll
    return R_combined

# Vertical street bounds
vertical_x1, vertical_x2 = -63650, 1450
vertical_y1, vertical_y2 = -75640, -70330

# Horizontal street bounds
horizontal_x1, horizontal_x2 = -33680, -28370
horizontal_y1, horizontal_y2 = -105550, -40420

# Intersection bounds
inter_x1 = max(vertical_x1, horizontal_x1)
inter_x2 = min(vertical_x2, horizontal_x2)
inter_y1 = max(vertical_y1, horizontal_y1)
inter_y2 = min(vertical_y2, horizontal_y2)

# Step 1: Generate full grid over the bounding box
x_min = min(vertical_x1, horizontal_x1)
x_max = max(vertical_x2, horizontal_x2)
y_min = min(vertical_y1, horizontal_y1)
y_max = max(vertical_y2, horizontal_y2)

x_vals = np.arange(x_min, x_max + grid_spacing, grid_spacing)
y_vals = np.arange(y_min, y_max + grid_spacing, grid_spacing)
xx, yy = np.meshgrid(x_vals, y_vals)
all_points = np.vstack((xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), z))).T

# Step 2: Filter points inside either street
in_vertical = (
    (all_points[:, 0] >= vertical_x1) & (all_points[:, 0] <= vertical_x2) &
    (all_points[:, 1] >= vertical_y1) & (all_points[:, 1] <= vertical_y2)
)
in_horizontal = (
    (all_points[:, 0] >= horizontal_x1) & (all_points[:, 0] <= horizontal_x2) &
    (all_points[:, 1] >= horizontal_y1) & (all_points[:, 1] <= horizontal_y2)
)

is_valid = in_vertical | in_horizontal
valid_points = all_points[is_valid]
rejected_points = all_points[~is_valid]

# Step 3: Center the grid within the road boundaries
x_min_valid, x_max_valid = valid_points[:, 0].min(), valid_points[:, 0].max()
y_min_valid, y_max_valid = valid_points[:, 1].min(), valid_points[:, 1].max()

street_x_min = min(vertical_x1, horizontal_x1)
street_x_max = max(vertical_x2, horizontal_x2)
street_y_min = min(vertical_y1, horizontal_y1)
street_y_max = max(vertical_y2, horizontal_y2)

# Avoid division by zero if valid points range is zero
x_range_valid = x_max_valid - x_min_valid
y_range_valid = y_max_valid - y_min_valid

x_offset = 0
if x_range_valid > 1e-6: # Check if range is non-zero
    x_offset = ((street_x_max - street_x_min) - x_range_valid) / 2 - (x_min_valid - street_x_min)

y_offset = 0
if y_range_valid > 1e-6: # Check if range is non-zero
    y_offset = ((street_y_max - street_y_min) - y_range_valid) / 2 - (y_min_valid - street_y_min)

shifted_valid_points = valid_points.astype(np.float64)
shifted_valid_points[:, 0] += x_offset
shifted_valid_points[:, 1] += y_offset

# Step 3b: Apply sorting logic before JSON export
is_intersection = in_horizontal & in_vertical
horizontal_mask = in_horizontal  # All horizontal points INCLUDING intersection
vertical_only_mask = in_vertical & ~is_intersection  # Only vertical points NOT in intersection

# Get original valid points belonging to each group
orig_horizontal_points = all_points[is_valid][horizontal_mask[is_valid]]
orig_vertical_only_points = all_points[is_valid][vertical_only_mask[is_valid]]

# Apply shift to horizontal points (including intersection)
shifted_horizontal_points = orig_horizontal_points.astype(np.float64)
shifted_horizontal_points[:, 0] += x_offset
shifted_horizontal_points[:, 1] += y_offset

# Apply shift to vertical-only points (excluding intersection)
shifted_vertical_only_points = orig_vertical_only_points.astype(np.float64)
shifted_vertical_only_points[:, 0] += x_offset
shifted_vertical_only_points[:, 1] += y_offset

# Calculate the center of the intersection (original coordinates)
intersection_center_orig = np.array([(inter_x1 + inter_x2) / 2, (inter_y1 + inter_y2) / 2, z])

# Shift the intersection center to be consistent with other points
intersection_center_shifted = intersection_center_orig.copy()
intersection_center_shifted[0] += x_offset
intersection_center_shifted[1] += y_offset

# Sort horizontal points: Y descending, then X descending
if shifted_horizontal_points.shape[0] > 0:
    sort_indices_h = np.lexsort((-shifted_horizontal_points[:, 0], -shifted_horizontal_points[:, 1]))
    sorted_horizontal = shifted_horizontal_points[sort_indices_h]
else:
    sorted_horizontal = np.empty((0, 3))

# Sort vertical-only points: Y descending (top to bottom)
if shifted_vertical_only_points.shape[0] > 0:
    # Changed to column-sweep: X ascending, then Y descending
    sort_indices_v = np.lexsort((-shifted_vertical_only_points[:, 1], shifted_vertical_only_points[:, 0]))
    sorted_vertical = shifted_vertical_only_points[sort_indices_v]
else:
    sorted_vertical = np.empty((0, 3))

# Combine sorted points with intersection center as transition point
if len(sorted_horizontal) > 0 and len(sorted_vertical) > 0:
    # Now we include intersection points in the horizontal path,
    # then add a transition point at intersection center,
    # then proceed to vertical-only points
    final_ordered_points = np.vstack((sorted_horizontal, [intersection_center_shifted], sorted_vertical))
elif len(sorted_horizontal) > 0:
    # Consider if intersection_center_shifted should be appended if no vertical points
    final_ordered_points = sorted_horizontal
elif len(sorted_vertical) > 0:
    # Consider if intersection_center_shifted should be prepended if no horizontal points
    final_ordered_points = sorted_vertical
else:
    final_ordered_points = np.empty((0, 3))

# --- Load Detection Data (if --detection flag is set) ---
detected_waypoint_indices = set() # Store 0-based indices from filenames
if args.detection:
    print("INFO: --detection flag is set. Loading detected image data...")
    # detection_files_paths = [
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-clearday-cam1/urban-clearday-cam1_detection_summary.json",
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-clearday-cam2/urban-clearday-cam2_detection_summary.json",
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-clearday-cam3/urban-clearday-cam3_detection_summary.json",
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-clearday-cam4/urban-clearday-cam4_detection_summary.json"
    # ]
    
    # detection_files_paths = [
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-cloudy-cam1/urban-cloudy-cam1_detection_summary.json",
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-cloudy-cam2/urban-cloudy-cam2_detection_summary.json",
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-cloudy-cam3/urban-cloudy-cam3_detection_summary.json",
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-cloudy-cam4/urban-cloudy-cam4_detection_summary.json"
    # ]
    
    # detection_files_paths = [
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-foggy-cam1/urban-foggy-cam1_detection_summary.json",
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-foggy-cam2/urban-foggy-cam2_detection_summary.json",
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-foggy-cam3/urban-foggy-cam3_detection_summary.json",
    #     "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-foggy-cam4/urban-foggy-cam4_detection_summary.json"
    # ]
    
    detection_files_paths = [
        "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-rainy-cam1/urban-rainy-cam1_detection_summary.json",
        "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-rainy-cam2/urban-rainy-cam2_detection_summary.json",
        "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-rainy-cam3/urban-rainy-cam3_detection_summary.json",
        "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban-processed/sensor-placement/urban-rainy-cam4/urban-rainy-cam4_detection_summary.json"
    ]

    for file_idx, file_path in enumerate(detection_files_paths):
        cam_id_for_log = f"camera{file_idx + 1}"
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f) # Renamed for clarity
                
                filenames_to_process = None
                if isinstance(data, dict) and "successful_detection_filenames" in data:
                    if isinstance(data["successful_detection_filenames"], list):
                        filenames_to_process = data["successful_detection_filenames"]
                    else:
                        print(f"WARNING: Key 'successful_detection_filenames' in {file_path} does not contain a list.")
                # Optional: Handle old format (direct list) if necessary, or remove for strictness
                # elif isinstance(data, list):
                #     filenames_to_process = data # Handles old format
                #     print(f"INFO: Processing {file_path} as a direct list of filenames (old format).")
                else:
                    print(f"WARNING: Expected a dictionary with 'successful_detection_filenames' key in {file_path}, but got {type(data)}.")

                if filenames_to_process:
                    loaded_count_for_file = 0
                    for filename in filenames_to_process:
                        if isinstance(filename, str) and '_' in filename:
                            try:
                                waypoint_idx_from_file = int(filename.split('_')[0])
                                # Filename prefix (e.g., 1) is assumed to be the 0-based index for final_ordered_points
                                detected_waypoint_indices.add(waypoint_idx_from_file)
                                loaded_count_for_file += 1
                            except ValueError:
                                print(f"WARNING: Could not parse index from filename '{filename}' in {file_path}")
                        else:
                            print(f"WARNING: Invalid filename format '{filename}' found in {file_path}")
                    if loaded_count_for_file > 0:
                        print(f"INFO: Loaded {loaded_count_for_file} detection entries from {cam_id_for_log} file: {file_path}")
                else:
                    print(f"INFO: No filenames found to process in {file_path} with the expected structure.")

            except json.JSONDecodeError:
                print(f"ERROR: Could not decode JSON from {file_path}")
            except Exception as e:
                print(f"ERROR: Reading or processing {file_path}: {e}")
        else:
            print(f"WARNING: Detection file not found for {cam_id_for_log}: {file_path}")
    
    if not detected_waypoint_indices:
        print("INFO: No detected waypoints loaded or files were empty/invalid.")
    else:
        print(f"INFO: Total unique detected waypoint indices to highlight: {len(detected_waypoint_indices)}")

# Step 4: Plot the result
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(shifted_valid_points[:, 0], shifted_valid_points[:, 1], c='blue', s=10, label='Shifted Grid Points')

# Highlight detected waypoints if detection mode is on
if args.detection and detected_waypoint_indices:
    detected_label_added = False
    # for idx in detected_waypoint_indices:
    #     if idx < len(final_ordered_points): # Ensure index is valid for final_ordered_points
    #         point = final_ordered_points[idx]
    #         ax.scatter(point[0], point[1], c='red', s=20, 
    #                    label='Detected Waypoint' if not detected_label_added else "", 
    for idx_from_file in detected_waypoint_indices: # idx_from_file is 0-based from the old system
        target_idx_for_plot = -1  # Initialize to an invalid index

        if idx_from_file < 113: # Old 0-indexed waypoints 0 through 112
            target_idx_for_plot = idx_from_file
        else: # Old 0-indexed waypoints 113 and onwards
            target_idx_for_plot = idx_from_file + 1

        if 0 <= target_idx_for_plot < len(final_ordered_points):
            point = final_ordered_points[target_idx_for_plot]
            ax.scatter(point[0], point[1], c='red', s=20,
                       label='Detected Waypoint' if not detected_label_added else "",
                       zorder=3) # zorder to draw on top
            detected_label_added = True
        else:
            print(f"WARNING: Detected waypoint index {idx_from_file} (from file) mapped to target {target_idx_for_plot}, which is out of bounds for final_ordered_points (len: {len(final_ordered_points)}). This point will not be highlighted.")

# Add number labels to points based on the final sorted order
for i, point in enumerate(final_ordered_points):
    ax.text(point[0] + 50, point[1] + 50, str(i + 1), fontsize=7, color='black')

# Draw road rectangles
ax.add_patch(patches.Rectangle((vertical_x1, vertical_y1), vertical_x2 - vertical_x1, vertical_y2 - vertical_y1,
                               edgecolor='green', facecolor='green', alpha=0.2, label='Vertical Street'))
ax.add_patch(patches.Rectangle((horizontal_x1, horizontal_y1), horizontal_x2 - horizontal_x1, horizontal_y2 - horizontal_y1,
                               edgecolor='red', facecolor='red', alpha=0.2, label='Horizontal Street'))

# Draw dashed intersection outline
ax.add_patch(patches.Rectangle((inter_x1, inter_y1), inter_x2 - inter_x1, inter_y2 - inter_y1,
                               edgecolor='purple', facecolor='none', linestyle='--', linewidth=2, label='Intersection'))

# Add buildings (based on the purple outlines in the image)
# Position buildings so their corners meet at the intersection
buildings = {
    "Building 1": {
        "x": inter_x1 - 35000,        # Left of intersection
        "y": inter_y2,                # Top of intersection
        "width": 35000,
        "height": 30000
    },
    "Building 2": {
        "x": inter_x2,                # Right of intersection
        "y": inter_y2,                # Top of intersection
        "width": 35000,
        "height": 30000
    },
    "Building 3": {
        "x": inter_x1 - 35000,        # Left of intersection
        "y": inter_y1 - 30000,        # Bottom of intersection
        "width": 35000,
        "height": 30000
    },
    "Building 4": {
        "x": inter_x2,                # Right of intersection
        "y": inter_y1 - 30000,        # Bottom of intersection
        "width": 35000,
        "height": 30000
    }
}

# Draw buildings
for name, building in buildings.items():
    # Draw the building rectangle
    building_rect = patches.Rectangle(
        (building["x"], building["y"]),
        building["width"],
        building["height"],
        edgecolor='gray',
        facecolor='gray',
        alpha=0.2,
        linewidth=2
    )
    ax.add_patch(building_rect)
    
    # Add building label in the center of the rectangle
    text_x = building["x"] + building["width"] / 2
    text_y = building["y"] + building["height"] / 2
    ax.text(text_x, text_y, name, fontsize=12, ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

# --- Draw Schematic FOV for each camera ---
for cam_id, camera in cameras.items():
    cam_xy = camera["position"][:2]
    cam_rotation_matrix = get_rotation_matrix(
        camera["orientation"]['pitch'],
        camera["orientation"]['yaw'],
        camera["orientation"]['roll']
    )
    
    # Get world forward direction (camera's local +X)
    local_forward = np.array([1, 0, 0])
    world_forward_3d = cam_rotation_matrix @ local_forward
    
    # Project onto XY plane
    world_forward_2d = world_forward_3d[:2]
    norm_2d = np.linalg.norm(world_forward_2d)
    
    if norm_2d > 1e-6:  # Check if camera is pointing mostly horizontally
        proj_forward_dir = world_forward_2d / norm_2d
        center_angle_rad = math.atan2(proj_forward_dir[1], proj_forward_dir[0])
        
        fov_h_rad = math.radians(camera["fov"])
        angle_left_rad = center_angle_rad + fov_h_rad / 2
        angle_right_rad = center_angle_rad - fov_h_rad / 2
        
        # Calculate points for the cone
        point_left_cone = cam_xy + schematic_fov_length * np.array([math.cos(angle_left_rad), math.sin(angle_left_rad)])
        point_right_cone = cam_xy + schematic_fov_length * np.array([math.cos(angle_right_rad), math.sin(angle_right_rad)])
        
        # Define polygon vertices for the cone
        fov_schematic_points = [cam_xy, point_left_cone, point_right_cone]
        
        # Get colors for this camera
        cone_color = camera_colors[cam_id]["cone"]
        line_color = camera_colors[cam_id]["line"]
        
        # Create and add the polygon patch with the specific color for this camera
        fov_schematic_polygon = patches.Polygon(
            fov_schematic_points,
            closed=True,
            edgecolor=cone_color,
            facecolor=cone_color,
            alpha=0.3,
            label=f'Schematic FOV ({cam_id})'
        )
        ax.add_patch(fov_schematic_polygon)
        
        # Calculate points for the extended lines
        point_left_extended = cam_xy + extended_fov_line_length * np.array([math.cos(angle_left_rad), math.sin(angle_left_rad)])
        point_right_extended = cam_xy + extended_fov_line_length * np.array([math.cos(angle_right_rad), math.sin(angle_right_rad)])
        
        # Plot the two extended FOV edge lines with the specific color for this camera
        ax.plot([cam_xy[0], point_left_extended[0]], [cam_xy[1], point_left_extended[1]],
                color=line_color, linestyle=':', linewidth=1.5, label=f'FOV Edge Lines ({cam_id})')
        ax.plot([cam_xy[0], point_right_extended[0]], [cam_xy[1], point_right_extended[1]],
                color=line_color, linestyle=':', linewidth=1.5)
        
    else:
        print(f"Warning: {cam_id} pointing too vertically, cannot draw schematic FOV.")
        # Draw a circle indicator if pointing vertically
        circle_color = camera_colors[cam_id]["cone"]
        fov_circle = patches.Circle(cam_xy, radius=schematic_fov_length/4, 
                                   edgecolor=circle_color, facecolor=circle_color, alpha=0.3, 
                                   label=f'Schematic FOV (Vertical) ({cam_id})')
        ax.add_patch(fov_circle)
    
    # Plot camera position with a specific marker color
    marker_color = camera_colors[cam_id]["cone"]  # Using the cone color for the marker
    ax.scatter(camera["position"][0], camera["position"][1], c=marker_color, s=100, marker='*', 
              label=f'{cam_id} Location')

# --- Final Plot Setup ---
ax.set_title('Sensor Placement and grid points')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.axis('equal')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# Step 5: Export to JSON using the sorted order
waypoint_list = []
for i, point in enumerate(final_ordered_points, start=1):
    waypoint = {
        "number": i,
        "x": float(point[0]),
        "y": float(point[1]),
        "altitude": float(point[2]),
        "speed": 5
    }
    waypoint_list.append(waypoint)

# Format the mission data matching the expected structure
mission_data = {
    "drone_config": {
        "coordinate_system": "xyz",
        "mission_file": "scripts/Data_collection/data_collection_config/generated_mission_data.json",
        "drone_type": "DJIS900"
    },
    "camera_config": {},
    "drones": [
        {
            "name": "Drone 1",
            "type": "Unspecified",
            "waypoints": waypoint_list
        }
    ],
    "cameras": [],
    "data_collection": {
        "base_output_dir": "D:/SiDG-ATRID-Dataset/Train_set/sidg-atrid-dataset-urban",
        "location": "brushify-urban",
        "weather_condition": "clear",
        "frame_rate": 10,
        "capture_duration": 100000,
        "max_images": 500,
        "check_exclusive_mask_color": True
    }
}

# Populate camera configurations
for cam_id, camera in cameras.items():
    # Add to camera_config section
    mission_data["camera_config"][cam_id] = {
        "location": {
            "x": float(camera["position"][0]),
            "y": float(camera["position"][1]),
            "z": float(camera["position"][2])
        },
        "rotation": {
            "pitch": float(camera["orientation"]["pitch"]),
            "yaw": float(camera["orientation"]["yaw"]),
            "roll": float(camera["orientation"]["roll"])
        }
    }
    
    # Add to cameras array with specs
    camera_entry = {
        cam_id: {
            "position": {
                "x": float(camera["position"][0]),
                "y": float(camera["position"][1]),
                "z": float(camera["position"][2])
            },
            "orientation": {
                "pitch": float(camera["orientation"]["pitch"]),
                "yaw": float(camera["orientation"]["yaw"]),
                "roll": float(camera["orientation"]["roll"])
            },
            "specs": {
                "resolution": {
                    "width": camera["resolution"]["width"],
                    "height": camera["resolution"]["height"]
                },
                "fov": float(camera["fov"])
            }
        }
    }
    mission_data["cameras"].append(camera_entry)

output_path = "scripts/Data_collection/data_collection_config/generated_mission_data_debug.json"
with open(output_path, 'w') as f:
    json.dump(mission_data, f, indent=2)

# Print summary
print(f"Total points generated (initial grid): {len(all_points)}")
print(f"Valid points kept and ordered: {len(final_ordered_points)}")
print(f"Points rejected from grid: {len(rejected_points)}")
print(f"Saved {len(final_ordered_points)} waypoints and {len(cameras)} cameras to {output_path}")
