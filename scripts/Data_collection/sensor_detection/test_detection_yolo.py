import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from tqdm import tqdm
import shutil
import argparse

# --- Configuration ---
BASE_INPUT_PATH = r"F:\testphotos\testrun\DJIS900"
MODEL_PATH = r"D:\SiDG-ATRID-Dataset\Model_test_results\Models\warsaw_dataset_model-yolo11\warsaw_dataset_model\weights\best.pt"
BASE_OUTPUT_PATH = r"F:\testphotos\testrun\DJIS900\detection_output"

Z_FOLDERS = ["brushify-urban-z1", "brushify-urban-z3"]
IOU_THRESHOLD = 0.25
CONFIDENCE_THRESHOLD = 0.25
DRONE_CLASS_ID = 0

# Default agent colors for mask processing (RGB yellow converted to BGR)
DEFAULT_AGENT_COLORS = {
    "drone": [0, 255, 255]  # RGB(255,255,0) in BGR format
}

# Predefined camera positions (from mission data)
CAMERA_POSITIONS = {
    1: {  # loc1
        "position": np.array([-74164.585036, -73303.671202, 773.166760]),
        "orientation": {'pitch': 7.0, 'yaw': 1.0, 'roll': 0.0}
    },
    2: {  # loc2  
        "position": np.array([-31023.204588, -30364.000000, 773.166760]),
        "orientation": {'pitch': 8.0, 'yaw': -90.0, 'roll': 0.0}
    },
    3: {  # loc3
        "position": np.array([-31333.000000, -108495.000000, 773.166760]),
        "orientation": {'pitch': 28.0, 'yaw': 86.601774, 'roll': 0.0}
    },
    4: {  # loc4
        "position": np.array([4244.677532, -73230.000000, 773.166760]),
        "orientation": {'pitch': 28.0, 'yaw': 177.0, 'roll': 0.0}
    }
}

# Camera colors for visualization
CAMERA_COLORS = {
    "camera1": {"cone": "cyan", "line": "navy"},
    "camera2": {"cone": "magenta", "line": "darkred"}, 
    "camera3": {"cone": "lime", "line": "green"},
    "camera4": {"cone": "orange", "line": "brown"}
}

# --- Helper Functions ---

def get_frame_index_from_filename(filename):
    """Extracts the leading integer (frame index) from a filename."""
    try:
        return int(filename.split('_')[0])
    except (ValueError, IndexError):
        return -1

def load_yolo_model(model_path):
    """Loads the YOLO model."""
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded YOLO model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        raise

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x_a = max(box1[0], box2[0])
    y_a = max(box1[1], box2[1])
    x_b = min(box1[2], box2[2])
    y_b = min(box1[3], box2[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    denominator = float(box1_area + box2_area - inter_area)
    if denominator == 0:
        return 0.0
    iou = inter_area / denominator
    return iou

def parse_yolo_label_file(label_path, img_width, img_height):
    """Parses a YOLO label file and converts to [x1, y1, x2, y2] format."""
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            try:
                class_id = int(parts[0])
                if class_id == DRONE_CLASS_ID:
                    x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts[1:])
                    
                    x_center = x_center_norm * img_width
                    y_center = y_center_norm * img_height
                    w = w_norm * img_width
                    h = h_norm * img_height
                    
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)
                    gt_boxes.append([x1, y1, x2, y2])
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping malformed line in {label_path}: '{line.strip()}' - {e}")
    return gt_boxes

def draw_bounding_box(image, box, color, label, thickness=2):
    """Draws a single bounding box on the image."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Adjust text placement if too close to top border
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20 
    cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

def process_mask_to_yolo(mask, agent_colors, img_shape):
    """Process a mask image to YOLO format labels"""
    labels = []
    bboxes = []
    
    height, width = mask.shape[:2]
    
    # Process each color in the agent_colors dictionary
    for agent_part, color in agent_colors.items():
        # Handle different color formats
        if isinstance(color, dict):
            # Extract RGB values and convert to BGR (since OpenCV uses BGR)
            bgr_color = [color.get('B', 0), color.get('G', 0), color.get('R', 0)]
        elif isinstance(color, list) and len(color) == 3:
            # Assume it's already in BGR format
            bgr_color = color
        else:
            print(f"Unsupported color format for {agent_part}: {color}")
            continue
        
        # Convert color to a NumPy array
        bgr_color = np.array(bgr_color, dtype=np.uint8)
        
        # Create a mask for exact color matching
        color_mask = np.all(mask == bgr_color, axis=2).astype(np.uint8) * 255
        
        # Check if any pixels match this color
        if np.any(color_mask):
            # Find contours in the binary mask
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (assuming it's the main object)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding box for the contour
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                
                # Assign class ID (0 for drone)
                class_id = 0
                
                label = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                
                labels.append(label)
                bboxes.append((x_center, y_center, w_norm, h_norm))
    
    return labels, bboxes

def process_subset_masks_to_bboxes(subset_path):
    """Process all mask images in a subset folder to create bounding box labels"""
    print(f"Processing masks in {subset_path}")
    
    # Check if subset has camera-specific subdirectories
    camera_dirs = [d for d in os.listdir(subset_path) if d.startswith('camera_') and os.path.isdir(os.path.join(subset_path, d))]
    
    if camera_dirs:
        # Handle camera-specific directory structure
        print(f"Found camera directories: {camera_dirs}")
        bbox_path = os.path.join(subset_path, 'bounding_box')
        os.makedirs(bbox_path, exist_ok=True)
        
        for camera_dir in camera_dirs:
            camera_path = os.path.join(subset_path, camera_dir)
            rgb_path = os.path.join(camera_path, 'rgb')
            mask_path = os.path.join(camera_path, 'mask')
            
            if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
                print(f"Warning: RGB or mask path missing in {camera_path}")
                continue
            
            # Check for agent color info in camera directory
            agent_info_path = os.path.join(camera_path, 'agent_color_info.json')
            if os.path.exists(agent_info_path):
                with open(agent_info_path, 'r') as f:
                    agent_colors = json.load(f)
            else:
                agent_colors = DEFAULT_AGENT_COLORS
                print(f"Using default agent colors for {camera_path}")
            
            # Get all RGB images from this camera
            rgb_files = [f for f in os.listdir(rgb_path) if f.endswith(('.png', '.jpg'))]
            
            for rgb_file in tqdm(rgb_files, desc=f"Processing {camera_dir}"):
                # Get base name without extension and remove "_lit" suffix
                base_name = os.path.splitext(rgb_file)[0]
                if base_name.endswith('_lit'):
                    base_name = base_name[:-4]
                
                # Look for corresponding mask file
                mask_file = f"{base_name}_object_mask.png"
                mask_file_path = os.path.join(mask_path, mask_file)
                
                if not os.path.exists(mask_file_path):
                    # Try alternative naming
                    mask_file = f"{base_name}.png"
                    mask_file_path = os.path.join(mask_path, mask_file)
                
                if not os.path.exists(mask_file_path):
                    print(f"Warning: No mask found for {rgb_file} in {camera_dir}")
                    continue
                
                # Read mask image
                mask = cv2.imread(mask_file_path)
                if mask is None:
                    print(f"Warning: Could not read mask {mask_file_path}")
                    continue
                
                # Read RGB image to get dimensions
                rgb_file_path = os.path.join(rgb_path, rgb_file)
                rgb_img = cv2.imread(rgb_file_path)
                if rgb_img is None:
                    print(f"Warning: Could not read RGB image {rgb_file_path}")
                    continue
                
                img_shape = rgb_img.shape
                
                # Process mask to get YOLO labels
                labels, bboxes = process_mask_to_yolo(mask, agent_colors, img_shape)
                
                # Save labels to file (combine all camera data into single bounding_box folder)
                label_file = f"{base_name}_lit.txt"
                label_file_path = os.path.join(bbox_path, label_file)
                
                with open(label_file_path, 'w') as f:
                    for label in labels:
                        f.write(label + '\n')
    else:
        # Handle direct rgb/mask directory structure (fallback)
        rgb_path = os.path.join(subset_path, 'rgb')
        mask_path = os.path.join(subset_path, 'mask')
        bbox_path = os.path.join(subset_path, 'bounding_box')
        
        # Create bounding_box directory
        os.makedirs(bbox_path, exist_ok=True)
        
        if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
            print(f"Warning: RGB or mask path missing in {subset_path}")
            return
        
        # Check for agent color info, use default if not found
        agent_info_path = os.path.join(subset_path, 'agent_color_info.json')
        if os.path.exists(agent_info_path):
            with open(agent_info_path, 'r') as f:
                agent_colors = json.load(f)
        else:
            agent_colors = DEFAULT_AGENT_COLORS
            print(f"Using default agent colors for {subset_path}")
        
        # Get all RGB images
        rgb_files = [f for f in os.listdir(rgb_path) if f.endswith(('.png', '.jpg'))]
        
        for rgb_file in tqdm(rgb_files, desc=f"Processing {os.path.basename(subset_path)}"):
            # Get base name without extension and remove "_lit" suffix
            base_name = os.path.splitext(rgb_file)[0]
            if base_name.endswith('_lit'):
                base_name = base_name[:-4]
            
            # Look for corresponding mask file
            mask_file = f"{base_name}_object_mask.png"
            mask_file_path = os.path.join(mask_path, mask_file)
            
            if not os.path.exists(mask_file_path):
                # Try alternative naming
                mask_file = f"{base_name}.png"
                mask_file_path = os.path.join(mask_path, mask_file)
            
            if not os.path.exists(mask_file_path):
                print(f"Warning: No mask found for {rgb_file}")
                continue
            
            # Read mask image
            mask = cv2.imread(mask_file_path)
            if mask is None:
                print(f"Warning: Could not read mask {mask_file_path}")
                continue
            
            # Read RGB image to get dimensions
            rgb_file_path = os.path.join(rgb_path, rgb_file)
            rgb_img = cv2.imread(rgb_file_path)
            if rgb_img is None:
                print(f"Warning: Could not read RGB image {rgb_file_path}")
                continue
            
            img_shape = rgb_img.shape
            
            # Process mask to get YOLO labels
            labels, bboxes = process_mask_to_yolo(mask, agent_colors, img_shape)
            
            # Save labels to file
            label_file = f"{base_name}_lit.txt"
            label_file_path = os.path.join(bbox_path, label_file)
            
            with open(label_file_path, 'w') as f:
                for label in labels:
                    f.write(label + '\n')

def load_subset_camera_config(z_folder_name, subset_name):
    """Load camera configuration for a specific subset"""
    # Map z_folder to config directory
    z_num = z_folder_name.split('-z')[1] if '-z' in z_folder_name else '1'
    config_path = f"scripts/Data_collection/data_collection_config/camera_config/z{z_num}/{subset_name}.json"
    
    if not os.path.exists(config_path):
        print(f"Warning: Camera config not found: {config_path}")
        return []
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return config_data
    except Exception as e:
        print(f"Error loading camera config from {config_path}: {e}")
        return []

def parse_camera_config(config_data):
    """Parse camera configuration array format to extract camera information
    
    Format: [camera_id, fov, resolution_width, resolution_height, exposure_level, focal_length, loc1, loc2, loc3, loc4, detection_output]
    """
    cameras = {}
    
    for cam_config in config_data:
        if len(cam_config) < 11:
            continue
            
        camera_id = cam_config[0]
        fov = cam_config[1]
        width = cam_config[2]
        height = cam_config[3]
        exposure = cam_config[4]
        focal_length = cam_config[5]
        loc1, loc2, loc3, loc4 = cam_config[6:10]
        
        # Determine camera position based on one-hot encoding
        location_id = None
        if loc1 == 1:
            location_id = 1
        elif loc2 == 1:
            location_id = 2
        elif loc3 == 1:
            location_id = 3
        elif loc4 == 1:
            location_id = 4
        
        if location_id and location_id in CAMERA_POSITIONS:
            cam_name = f"camera{camera_id}"
            cameras[cam_name] = {
                "camera_id": camera_id,
                "position": CAMERA_POSITIONS[location_id]["position"],
                "orientation": CAMERA_POSITIONS[location_id]["orientation"],
                "fov": fov,
                "resolution": {"width": width, "height": height},
                "exposure": exposure,
                "focal_length": focal_length,
                "location_id": location_id
            }
    
    return cameras

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
    R_combined = R_yaw @ R_pitch @ R_roll
    return R_combined

def generate_subset_visualization(subset_name, subset_detection_data, z_folder_name, output_path):
    """Generate waypoint visualization for a single subset with its specific camera configuration"""
    
    # Load subset-specific camera configuration
    camera_config_data = load_subset_camera_config(z_folder_name, subset_name)
    if not camera_config_data:
        print(f"Warning: Could not load camera config for {subset_name}, skipping visualization")
        return
    
    cameras = parse_camera_config(camera_config_data)
    if not cameras:
        print(f"Warning: No valid cameras found in config for {subset_name}, skipping visualization")
        return
    
    print(f"Generating visualization for {subset_name} with {len(cameras)} cameras")
    
    # Grid parameters (matching waypoint_gen.py)
    z = 260
    grid_spacing = 3200
    
    # Street bounds
    vertical_x1, vertical_x2 = -63650, 1450
    vertical_y1, vertical_y2 = -75640, -70330
    horizontal_x1, horizontal_x2 = -33680, -28370
    horizontal_y1, horizontal_y2 = -105550, -40420
    
    # Generate grid points (simplified version from waypoint_gen.py)
    vertical_width = vertical_x2 - vertical_x1
    vertical_height = vertical_y2 - vertical_y1
    vertical_center_x = (vertical_x1 + vertical_x2) / 2
    vertical_center_y = (vertical_y1 + vertical_y2) / 2

    vertical_cols = int(vertical_width / grid_spacing) + 1
    vertical_rows = int(vertical_height / grid_spacing) + 1
    vertical_x_start = vertical_center_x - (vertical_cols - 1) * grid_spacing / 2
    vertical_y_start = vertical_center_y - (vertical_rows - 1) * grid_spacing / 2

    vertical_x_vals = np.array([vertical_x_start + i * grid_spacing for i in range(vertical_cols)])
    vertical_y_vals = np.array([vertical_y_start + i * grid_spacing for i in range(vertical_rows)])
    vertical_xx, vertical_yy = np.meshgrid(vertical_x_vals, vertical_y_vals)
    vertical_points = np.vstack((vertical_xx.ravel(), vertical_yy.ravel(), np.full_like(vertical_xx.ravel(), z))).T

    horizontal_width = horizontal_x2 - horizontal_x1
    horizontal_height = horizontal_y2 - horizontal_y1
    horizontal_center_x = (horizontal_x1 + horizontal_x2) / 2
    horizontal_center_y = (horizontal_y1 + horizontal_y2) / 2

    horizontal_cols = int(horizontal_width / grid_spacing) + 1
    horizontal_rows = int(horizontal_height / grid_spacing) + 1
    horizontal_x_start = horizontal_center_x - (horizontal_cols - 1) * grid_spacing / 2
    horizontal_y_start = horizontal_center_y - (horizontal_rows - 1) * grid_spacing / 2

    horizontal_x_vals = np.array([horizontal_x_start + i * grid_spacing for i in range(horizontal_cols)])
    horizontal_y_vals = np.array([horizontal_y_start + i * grid_spacing for i in range(horizontal_rows)])
    horizontal_xx, horizontal_yy = np.meshgrid(horizontal_x_vals, horizontal_y_vals)
    horizontal_points = np.vstack((horizontal_xx.ravel(), horizontal_yy.ravel(), np.full_like(horizontal_xx.ravel(), z))).T

    # Combine and filter points
    all_points = np.vstack((vertical_points, horizontal_points))
    
    in_vertical = (
        (all_points[:, 0] >= vertical_x1) & (all_points[:, 0] <= vertical_x2) &
        (all_points[:, 1] >= vertical_y1) & (all_points[:, 1] <= vertical_y2)
    )
    in_horizontal = (
        (all_points[:, 0] >= horizontal_x1) & (all_points[:, 0] <= horizontal_x2) &
        (all_points[:, 1] >= horizontal_y1) & (all_points[:, 1] <= horizontal_y2)
    )

    is_valid = in_vertical | in_horizontal
    valid_points = all_points[is_valid].astype(np.float64)
    
    # Extract detected waypoint indices from subset data
    detected_indices = set()
    if isinstance(subset_detection_data, dict):
        # Check if this is a total summary (has subset_name key)
        if 'subset_name' in subset_detection_data and 'successful_detection_indices' in subset_detection_data:
            # This is a total summary
            detected_indices.update(subset_detection_data['successful_detection_indices'])
        else:
            # This is individual camera data
            for camera_data in subset_detection_data.values():
                if isinstance(camera_data, dict) and 'successful_detection_indices' in camera_data:
                    detected_indices.update(camera_data['successful_detection_indices'])
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot all waypoints
    ax.scatter(valid_points[:, 0], valid_points[:, 1], c='blue', s=10, label='Waypoints', alpha=0.6)
    
    # Highlight detected waypoints
    if detected_indices:
        detected_label_added = False
        for idx in detected_indices:
            if 0 <= idx < len(valid_points):
                point = valid_points[idx]
                ax.scatter(point[0], point[1], c='red', s=30,
                          label='Detected Waypoint' if not detected_label_added else "",
                          zorder=3)
                detected_label_added = True
    
    # Draw street rectangles
    ax.add_patch(patches.Rectangle((vertical_x1, vertical_y1), vertical_x2 - vertical_x1, vertical_y2 - vertical_y1,
                                  edgecolor='green', facecolor='green', alpha=0.2, label='Vertical Street'))
    ax.add_patch(patches.Rectangle((horizontal_x1, horizontal_y1), horizontal_x2 - horizontal_x1, horizontal_y2 - horizontal_y1,
                                  edgecolor='red', facecolor='red', alpha=0.2, label='Horizontal Street'))
    
    # Draw camera FOVs and positions
    schematic_fov_length = 5 * grid_spacing
    extended_fov_line_length = 30 * grid_spacing
    
    # Camera colors based on camera ID
    camera_colors_map = {
        "camera1": {"cone": "cyan", "line": "navy"},
        "camera2": {"cone": "magenta", "line": "darkred"}, 
        "camera3": {"cone": "lime", "line": "green"},
        "camera4": {"cone": "orange", "line": "brown"}
    }
    
    for cam_id, camera in cameras.items():
        cam_xy = camera["position"][:2]
        cam_rotation_matrix = get_rotation_matrix(
            camera["orientation"]['pitch'],
            camera["orientation"]['yaw'],
            camera["orientation"]['roll']
        )
        
        # Get world forward direction
        local_forward = np.array([1, 0, 0])
        world_forward_3d = cam_rotation_matrix @ local_forward
        world_forward_2d = world_forward_3d[:2]
        norm_2d = np.linalg.norm(world_forward_2d)
        
        if norm_2d > 1e-6:
            proj_forward_dir = world_forward_2d / norm_2d
            center_angle_rad = math.atan2(proj_forward_dir[1], proj_forward_dir[0])
            
            fov_h_rad = math.radians(camera["fov"])
            angle_left_rad = center_angle_rad + fov_h_rad / 2
            angle_right_rad = center_angle_rad - fov_h_rad / 2
            
            # Calculate FOV cone points
            point_left_cone = cam_xy + schematic_fov_length * np.array([math.cos(angle_left_rad), math.sin(angle_left_rad)])
            point_right_cone = cam_xy + schematic_fov_length * np.array([math.cos(angle_right_rad), math.sin(angle_right_rad)])
            
            fov_schematic_points = [cam_xy, point_left_cone, point_right_cone]
            cone_color = camera_colors_map.get(cam_id, {"cone": "gray", "line": "black"})["cone"]
            line_color = camera_colors_map.get(cam_id, {"cone": "gray", "line": "black"})["line"]
            
            # Draw FOV cone
            fov_schematic_polygon = patches.Polygon(
                fov_schematic_points,
                closed=True,
                edgecolor=cone_color,
                facecolor=cone_color,
                alpha=0.3
            )
            ax.add_patch(fov_schematic_polygon)
            
            # Draw extended FOV lines
            point_left_extended = cam_xy + extended_fov_line_length * np.array([math.cos(angle_left_rad), math.sin(angle_left_rad)])
            point_right_extended = cam_xy + extended_fov_line_length * np.array([math.cos(angle_right_rad), math.sin(angle_right_rad)])
            
            ax.plot([cam_xy[0], point_left_extended[0]], [cam_xy[1], point_left_extended[1]],
                    color=line_color, linestyle=':', linewidth=1.5, 
                    label=f'FOV {cam_id} (FOV:{camera["fov"]}°)' if list(cameras.keys()).index(cam_id) == 0 else "")
            ax.plot([cam_xy[0], point_right_extended[0]], [cam_xy[1], point_right_extended[1]],
                    color=line_color, linestyle=':', linewidth=1.5)
        
        # Plot camera position
        marker_color = camera_colors_map.get(cam_id, {"cone": "gray"})["cone"]
        ax.scatter(camera["position"][0], camera["position"][1], c=marker_color, s=100, marker='*', 
                  label=f'{cam_id} (Loc{camera["location_id"]})' if list(cameras.keys()).index(cam_id) == 0 else "")
    
    # Add camera info text
    camera_info_text = []
    for cam_id, camera in cameras.items():
        camera_info_text.append(f"{cam_id}: FOV {camera['fov']}° at Location {camera['location_id']}")
    
    ax.text(0.02, 0.98, '\n'.join(camera_info_text), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Determine title based on data type
    if isinstance(subset_detection_data, dict) and 'subset_name' in subset_detection_data:
        # This is a total summary
        total_cameras = subset_detection_data.get('total_cameras', 0)
        ax.set_title(f'Detection Visualization - {z_folder_name} - {subset_name}\nCombined: {len(detected_indices)} waypoints detected across {total_cameras} cameras')
    else:
        # This is individual camera data
        ax.set_title(f'Detection Visualization - {z_folder_name} - {subset_name}\nDetected: {len(detected_indices)} waypoints')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_path, f'{subset_name}_detection_visualization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved subset visualization to {plot_path}")

def test_camera_with_yolo(camera_path, camera_name, labels_dir, model, output_camera_dir):
    """Test a single camera with YOLO model"""
    rgb_path = os.path.join(camera_path, "rgb")
    
    if not os.path.exists(rgb_path):
        print(f"Warning: RGB directory missing in {camera_path}")
        return None
    
    # Create output directories for this camera
    output_debug_images_dir = os.path.join(output_camera_dir, "debug_images")
    os.makedirs(output_debug_images_dir, exist_ok=True)
    
    successful_detection_filenames = []
    successful_detection_indices = []
    detection_status_all_images = []
    
    # Get all image files and sort them
    all_files_in_dir = [f for f in os.listdir(rgb_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(all_files_in_dir, key=get_frame_index_from_filename)
    
    for idx, image_file_name in enumerate(tqdm(image_files, desc=f"Testing {camera_name}")):
        image_path = os.path.join(rgb_path, image_file_name)
        
        # Determine label file name
        image_stem = Path(image_file_name).stem
        if image_stem.endswith('_lit'):
            label_base_name = image_stem[:-4]
        else:
            label_base_name = image_stem
        
        label_file_name = f"{label_base_name}_lit.txt"
        label_path = os.path.join(labels_dir, label_file_name)
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Parse ground truth labels
        gt_boxes = parse_yolo_label_file(label_path, img_width, img_height)
        
        # Run YOLO prediction
        results = model(img, conf=CONFIDENCE_THRESHOLD, classes=[DRONE_CLASS_ID], verbose=False)
        
        pred_boxes = []
        if results and results[0].boxes:
            for box_data in results[0].boxes:
                xyxy = box_data.xyxy[0].tolist()
                pred_boxes.append(xyxy)
        
        # Matching logic (same as yolo_test.py)
        gt_match_status = [{'detected': False, 'iou': 0.0} for _ in gt_boxes]
        pred_match_status = [{'is_tp': False, 'iou': 0.0} for _ in pred_boxes]
        
        available_gt_indices = list(range(len(gt_boxes)))
        
        for i, pred_b in enumerate(pred_boxes):
            best_iou_for_this_pred = -1
            best_gt_idx_for_this_pred = -1
            best_available_gt_local_idx = -1

            for local_gt_idx, original_gt_idx in enumerate(available_gt_indices):
                gt_b = gt_boxes[original_gt_idx]
                iou = calculate_iou(pred_b, gt_b)
                if iou > best_iou_for_this_pred:
                    best_iou_for_this_pred = iou
                    best_gt_idx_for_this_pred = original_gt_idx
                    best_available_gt_local_idx = local_gt_idx
            
            if best_iou_for_this_pred >= IOU_THRESHOLD:
                pred_match_status[i]['is_tp'] = True
                pred_match_status[i]['iou'] = best_iou_for_this_pred
                
                if not gt_match_status[best_gt_idx_for_this_pred]['detected'] or \
                   best_iou_for_this_pred > gt_match_status[best_gt_idx_for_this_pred]['iou']:
                    gt_match_status[best_gt_idx_for_this_pred]['detected'] = True
                    gt_match_status[best_gt_idx_for_this_pred]['iou'] = best_iou_for_this_pred
                
                available_gt_indices.pop(best_available_gt_local_idx)
        
        image_has_any_successful_detection = any(gs['detected'] for gs in gt_match_status)
        
        # Determine debug image suffix based on detection results
        if image_has_any_successful_detection:
            successful_detection_filenames.append(image_file_name)
            parsed_frame_index = get_frame_index_from_filename(image_file_name)
            if parsed_frame_index != -1:
                successful_detection_indices.append(parsed_frame_index)
            else:
                successful_detection_indices.append(idx)
            detection_status_all_images.append(1)
            debug_image_suffix = "_debug.jpg"
        elif gt_boxes and not image_has_any_successful_detection:
            detection_status_all_images.append(0)
            debug_image_suffix = "_FN.jpg"
        elif not gt_boxes and pred_boxes:
            detection_status_all_images.append(0)
            debug_image_suffix = "_FP.jpg"
        else:
            detection_status_all_images.append(0)
            debug_image_suffix = "_no_detection.jpg"
        
        # Save debug image for ALL images
        debug_img = img.copy()
        
        # Draw Ground Truth Boxes
        for j, gt_b in enumerate(gt_boxes):
            status = gt_match_status[j]
            if status['detected']:
                color = (0, 255, 0)  # Green
                label = f"GT (Detected IoU:{status['iou']:.2f})"
            else:
                color = (0, 165, 255)  # Orange
                label = "GT (Missed)"
            draw_bounding_box(debug_img, gt_b, color, label)
        
        # Draw Predicted Boxes
        for i, pred_b in enumerate(pred_boxes):
            status = pred_match_status[i]
            if status['is_tp']:
                color = (255, 0, 0)  # Blue
                label = f"Pred (TP IoU:{status['iou']:.2f})"
            else:
                color = (0, 0, 255)  # Red
                label = "Pred (FP)"
            draw_bounding_box(debug_img, pred_b, color, label)
        
        debug_image_filename = f"{Path(image_file_name).stem}{debug_image_suffix}"
        debug_image_path = os.path.join(output_debug_images_dir, debug_image_filename)
        cv2.imwrite(debug_image_path, debug_img)
    
    # Save detection summary for this camera
    output_data = {
        "camera_name": camera_name,
        "successful_detection_filenames": successful_detection_filenames,
        "successful_detection_indices": successful_detection_indices,
        "detection_status_all_images": detection_status_all_images
    }
    
    output_json_path = os.path.join(output_camera_dir, f"{camera_name}_detection_summary.json")
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Camera {camera_name}: Processed {len(image_files)} images, {len(successful_detection_filenames)} successful detections")
    return output_data

def test_camera_with_yolo_no_debug(camera_path, camera_name, labels_dir, model):
    """Test a single camera with YOLO model without saving debug images"""
    rgb_path = os.path.join(camera_path, "rgb")
    
    if not os.path.exists(rgb_path):
        print(f"Warning: RGB directory missing in {camera_path}")
        return None
    
    successful_detection_filenames = []
    successful_detection_indices = []
    detection_status_all_images = []
    
    # Get all image files and sort them
    all_files_in_dir = [f for f in os.listdir(rgb_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(all_files_in_dir, key=get_frame_index_from_filename)
    
    for idx, image_file_name in enumerate(tqdm(image_files, desc=f"Testing {camera_name}")):
        image_path = os.path.join(rgb_path, image_file_name)
        
        # Determine label file name
        image_stem = Path(image_file_name).stem
        if image_stem.endswith('_lit'):
            label_base_name = image_stem[:-4]
        else:
            label_base_name = image_stem
        
        label_file_name = f"{label_base_name}_lit.txt"
        label_path = os.path.join(labels_dir, label_file_name)
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Parse ground truth labels
        gt_boxes = parse_yolo_label_file(label_path, img_width, img_height)
        
        # Run YOLO prediction
        results = model(img, conf=CONFIDENCE_THRESHOLD, classes=[DRONE_CLASS_ID], verbose=False)
        
        pred_boxes = []
        if results and results[0].boxes:
            for box_data in results[0].boxes:
                xyxy = box_data.xyxy[0].tolist()
                pred_boxes.append(xyxy)
        
        # Matching logic (same as yolo_test.py)
        gt_match_status = [{'detected': False, 'iou': 0.0} for _ in gt_boxes]
        pred_match_status = [{'is_tp': False, 'iou': 0.0} for _ in pred_boxes]
        
        available_gt_indices = list(range(len(gt_boxes)))
        
        for i, pred_b in enumerate(pred_boxes):
            best_iou_for_this_pred = -1
            best_gt_idx_for_this_pred = -1
            best_available_gt_local_idx = -1

            for local_gt_idx, original_gt_idx in enumerate(available_gt_indices):
                gt_b = gt_boxes[original_gt_idx]
                iou = calculate_iou(pred_b, gt_b)
                if iou > best_iou_for_this_pred:
                    best_iou_for_this_pred = iou
                    best_gt_idx_for_this_pred = original_gt_idx
                    best_available_gt_local_idx = local_gt_idx
            
            if best_iou_for_this_pred >= IOU_THRESHOLD:
                pred_match_status[i]['is_tp'] = True
                pred_match_status[i]['iou'] = best_iou_for_this_pred
                
                if not gt_match_status[best_gt_idx_for_this_pred]['detected'] or \
                   best_iou_for_this_pred > gt_match_status[best_gt_idx_for_this_pred]['iou']:
                    gt_match_status[best_gt_idx_for_this_pred]['detected'] = True
                    gt_match_status[best_gt_idx_for_this_pred]['iou'] = best_iou_for_this_pred
                
                available_gt_indices.pop(best_available_gt_local_idx)
        
        image_has_any_successful_detection = any(gs['detected'] for gs in gt_match_status)
        
        # Track detection results (no debug images saved)
        if image_has_any_successful_detection:
            successful_detection_filenames.append(image_file_name)
            parsed_frame_index = get_frame_index_from_filename(image_file_name)
            if parsed_frame_index != -1:
                successful_detection_indices.append(parsed_frame_index)
            else:
                successful_detection_indices.append(idx)
            detection_status_all_images.append(1)
        else:
            detection_status_all_images.append(0)
    
    # Return detection summary for this camera (no file saved in standard mode)
    output_data = {
        "camera_name": camera_name,
        "successful_detection_filenames": successful_detection_filenames,
        "successful_detection_indices": successful_detection_indices,
        "detection_status_all_images": detection_status_all_images
    }
    
    print(f"Camera {camera_name}: Processed {len(image_files)} images, {len(successful_detection_filenames)} successful detections")
    return output_data

def test_subset_with_yolo(subset_path, model, z_folder_name, debug_mode=True):
    """Test a single subset with YOLO model, processing each camera individually"""
    print(f"\nTesting subset: {os.path.basename(subset_path)}")
    
    # Check if subset has camera-specific subdirectories
    camera_dirs = [d for d in os.listdir(subset_path) if d.startswith('camera_') and os.path.isdir(os.path.join(subset_path, d))]
    labels_dir = os.path.join(subset_path, "bounding_box")
    
    if not os.path.exists(labels_dir):
        print(f"Warning: Labels directory missing in {subset_path}")
        return None
    
    if not camera_dirs:
        print(f"Warning: No camera directories found in {subset_path}")
        return None
    
    subset_name = os.path.basename(subset_path)
    
    if debug_mode:
        # Create output directory for this subset (debug mode)
        output_subset_dir = os.path.join(BASE_OUTPUT_PATH, z_folder_name, subset_name)
        os.makedirs(output_subset_dir, exist_ok=True)
        
        # Copy subset configuration JSON
        z_num = z_folder_name.split('-z')[1] if '-z' in z_folder_name else '1'
        subset_config_source = f"scripts/Data_collection/data_collection_config/camera_config/z{z_num}/{subset_name}.json"
        subset_config_dest = os.path.join(output_subset_dir, f"{subset_name}.json")
        
        if os.path.exists(subset_config_source):
            shutil.copy2(subset_config_source, subset_config_dest)
            print(f"Copied subset config to {subset_config_dest}")
        else:
            print(f"Warning: Subset config not found: {subset_config_source}")
    else:
        # Standard mode - only copy subset config to main output directory
        z_num = z_folder_name.split('-z')[1] if '-z' in z_folder_name else '1'
        subset_config_source = f"scripts/Data_collection/data_collection_config/camera_config/z{z_num}/{subset_name}.json"
        subset_config_dest = os.path.join(BASE_OUTPUT_PATH, z_folder_name, f"{subset_name}.json")
        
        if os.path.exists(subset_config_source):
            shutil.copy2(subset_config_source, subset_config_dest)
            print(f"Copied subset config to {subset_config_dest}")
        else:
            print(f"Warning: Subset config not found: {subset_config_source}")
        
        output_subset_dir = None  # No individual subset directory in standard mode
    
    subset_detection_data = {}
    
    # Process each camera individually
    for camera_dir in sorted(camera_dirs):
        camera_path = os.path.join(subset_path, camera_dir)
        
        if debug_mode:
            output_camera_dir = os.path.join(output_subset_dir, camera_dir)
            os.makedirs(output_camera_dir, exist_ok=True)
            
            # Copy camera config JSON if it exists
            camera_config_source = os.path.join(camera_path, "camera_config.json")
            camera_config_dest = os.path.join(output_camera_dir, "camera_config.json")
            
            if os.path.exists(camera_config_source):
                shutil.copy2(camera_config_source, camera_config_dest)
                print(f"Copied camera config to {camera_config_dest}")
            else:
                print(f"Warning: Camera config not found: {camera_config_source}")
            
            # Test this camera with YOLO (with debug images)
            camera_results = test_camera_with_yolo(camera_path, camera_dir, labels_dir, model, output_camera_dir)
        else:
            # Test this camera with YOLO (without debug images)
            camera_results = test_camera_with_yolo_no_debug(camera_path, camera_dir, labels_dir, model)
        
        if camera_results:
            subset_detection_data[camera_dir] = camera_results
    
    if debug_mode:
        # Create combined subset summary
        combined_successful_filenames = []
        combined_successful_indices = set()
        
        # Collect all unique successful detections across cameras
        for camera_name, camera_data in subset_detection_data.items():
            if isinstance(camera_data, dict):
                # Add filenames (keep all, even if duplicated across cameras)
                combined_successful_filenames.extend(camera_data.get('successful_detection_filenames', []))
                # Add indices (will be deduplicated by set)
                combined_successful_indices.update(camera_data.get('successful_detection_indices', []))
        
        # Convert back to sorted list
        combined_successful_indices_list = sorted(list(combined_successful_indices))
        
        # Create detection status for all waypoints/images
        # We need to determine the total number of waypoints/images processed
        max_images = 0
        for camera_name, camera_data in subset_detection_data.items():
            if isinstance(camera_data, dict):
                camera_total = len(camera_data.get('detection_status_all_images', []))
                max_images = max(max_images, camera_total)
        
        # Create combined detection status array
        combined_detection_status = [0] * max_images
        for idx in combined_successful_indices_list:
            if 0 <= idx < max_images:
                combined_detection_status[idx] = 1
        
        # Create subset summary
        subset_summary = {
            "subset_name": subset_name,
            "total_cameras": len(camera_dirs),
            "successful_detection_filenames": combined_successful_filenames,
            "successful_detection_indices": combined_successful_indices_list,
            "detection_status_all_images": combined_detection_status,
            "total_unique_detections": len(combined_successful_indices_list),
            "total_images_per_camera": max_images,
            "camera_breakdown": {camera: len(data.get('successful_detection_indices', [])) 
                               for camera, data in subset_detection_data.items() if isinstance(data, dict)}
        }
        
        # Save subset summary
        subset_summary_path = os.path.join(output_subset_dir, f"{subset_name}_total_summary.json")
        with open(subset_summary_path, 'w') as f:
            json.dump(subset_summary, f, indent=4)
        
        print(f"Saved subset total summary to {subset_summary_path}")
        print(f"Subset {subset_name}: {len(combined_successful_indices_list)} unique waypoints detected across {len(camera_dirs)} cameras")
        
        # Update subset config with detection results
        update_subset_config_with_detection_results(subset_config_dest, subset_detection_data)
    else:
        # Standard mode - just update the subset config with detection results
        subset_config_dest = os.path.join(BASE_OUTPUT_PATH, z_folder_name, f"{subset_name}.json")
        update_subset_config_with_detection_results(subset_config_dest, subset_detection_data)
    
    print(f"Completed testing subset {subset_name} with {len(camera_dirs)} cameras")
    return subset_detection_data

def update_subset_config_with_detection_results(subset_config_path, camera_detection_data):
    """Update the subset configuration file with detection results"""
    try:
        # Load the subset configuration
        with open(subset_config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create a mapping from camera_id to detection results
        camera_results_map = {}
        for camera_name, camera_data in camera_detection_data.items():
            if isinstance(camera_data, dict) and 'detection_status_all_images' in camera_data:
                # Extract camera_id from camera_name (e.g., "camera_1" -> 1)
                try:
                    camera_id = int(camera_name.split('_')[1])
                    camera_results_map[camera_id] = camera_data['detection_status_all_images']
                except (ValueError, IndexError):
                    print(f"Warning: Could not extract camera ID from {camera_name}")
        
        # Update each camera configuration with its detection results
        for i, cam_config in enumerate(config_data):
            if len(cam_config) >= 11:
                camera_id = cam_config[0]
                if camera_id in camera_results_map:
                    # Replace the empty list with detection results
                    cam_config[10] = camera_results_map[camera_id]
                    print(f"Updated camera {camera_id} with {len(camera_results_map[camera_id])} detection results")
                else:
                    print(f"Warning: No detection results found for camera {camera_id}")
        
        # Save the updated configuration
        with open(subset_config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Updated subset config with detection results: {subset_config_path}")
        
    except Exception as e:
        print(f"Error updating subset config {subset_config_path}: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO Detection Testing with Camera Configuration')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with full output (debug images, individual camera folders, visualizations)')
    args = parser.parse_args()
    
    print("Starting YOLO detection testing...")
    if args.debug:
        print("Debug mode enabled - Full output with debug images and visualizations")
    else:
        print("Standard mode - Only subset configs with detection results")
    
    # Load YOLO model
    model = load_yolo_model(MODEL_PATH)
    
    # Create output directory
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    
    for z_folder in Z_FOLDERS:
        print(f"\n{'='*50}")
        print(f"Processing {z_folder}")
        print(f"{'='*50}")
        
        z_folder_path = os.path.join(BASE_INPUT_PATH, z_folder)
        z_output_path = os.path.join(BASE_OUTPUT_PATH, z_folder)
        
        if not os.path.exists(z_folder_path):
            print(f"Warning: Z folder not found: {z_folder_path}")
            continue
        
        os.makedirs(z_output_path, exist_ok=True)
        
        # Find all subset folders
        subset_folders = [d for d in os.listdir(z_folder_path) 
                         if d.startswith('subset_') and os.path.isdir(os.path.join(z_folder_path, d))]
        subset_folders.sort()
        
        print(f"Found {len(subset_folders)} subset folders: {subset_folders}")
        
        detection_data = {}
        
        for subset_folder in subset_folders:
            subset_path = os.path.join(z_folder_path, subset_folder)
            
            # Step 1: Process masks to create bounding boxes
            process_subset_masks_to_bboxes(subset_path)
            
            # Step 2: Test with YOLO model
            subset_results = test_subset_with_yolo(subset_path, model, z_folder, args.debug)
            if subset_results:
                detection_data[subset_folder] = subset_results
        
        if args.debug:
            # Step 3: Generate subset-specific visualizations using total summary
            for subset_folder, subset_results in detection_data.items():
                if subset_results:
                    # Load the total summary for visualization
                    subset_summary_path = os.path.join(z_output_path, subset_folder, f"{subset_folder}_total_summary.json")
                    if os.path.exists(subset_summary_path):
                        with open(subset_summary_path, 'r') as f:
                            total_summary = json.load(f)
                        generate_subset_visualization(subset_folder, total_summary, z_folder, z_output_path)
                    else:
                        print(f"Warning: Total summary not found for {subset_folder}, using individual camera data")
                        generate_subset_visualization(subset_folder, subset_results, z_folder, z_output_path)
            
            # Step 4: Save combined detection results
            combined_successful_indices = []
            total_cameras = 0
            total_images = 0
            
            for subset_name, subset_data in detection_data.items():
                if isinstance(subset_data, dict):
                    for camera_name, camera_data in subset_data.items():
                        if isinstance(camera_data, dict) and 'successful_detection_indices' in camera_data:
                            combined_successful_indices.extend(camera_data['successful_detection_indices'])
                            total_cameras += 1
                            total_images += len(camera_data.get('detection_status_all_images', []))
            
            combined_results = {
                "z_folder": z_folder,
                "total_subsets": len(subset_folders),
                "total_cameras": total_cameras,
                "total_images_processed": total_images,
                "successful_detection_indices": sorted(list(set(combined_successful_indices))),
                "total_successful_detections": len(set(combined_successful_indices)),
                "subset_results": detection_data
            }
            
            combined_results_path = os.path.join(z_output_path, f"{z_folder}_combined_results.json")
            with open(combined_results_path, 'w') as f:
                json.dump(combined_results, f, indent=4)
            
            print(f"\nCompleted processing {z_folder}")
            print(f"Total unique successful detections: {len(set(combined_successful_indices))}")
            print(f"Results saved to: {z_output_path}")
        else:
            print(f"\nCompleted processing {z_folder} in standard mode")
            print(f"Subset configs with detection results saved to: {z_output_path}")

if __name__ == "__main__":
    main()
