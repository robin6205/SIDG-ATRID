import os
import json
import math
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import shutil
import argparse
import yaml
from collections import defaultdict

# Canonical class mapping for multi-class detection
BASE_CLASS_MAPPING = {
    'quadcopter': 0,
    'kraken': 1,
    'opterra': 2,
    'lrquad': 3,
    'unknown_drone': 4
}

# Aliases to map various drone type strings to canonical class names
DRONE_TYPE_ALIASES = {
    'djis900': 'quadcopter',
    'dji_s900': 'quadcopter',
    'dji-s900': 'quadcopter',
    'djis-900': 'quadcopter',
    'dji s900': 'quadcopter',
    'djifpv': 'quadcopter',
    'dji_fpv': 'quadcopter',
    'dji fpv': 'quadcopter',
    'djimini2': 'quadcopter',
    'dji_mini2': 'quadcopter',
    'dji mini2': 'quadcopter',
    'djimini': 'quadcopter',
    'dji_mini': 'quadcopter',
    'dji mini': 'quadcopter',
    'djiphantom': 'quadcopter',
    'dji_phantom': 'quadcopter',
    'dji phantom': 'quadcopter',
    'dji_phantom4': 'quadcopter',
    'dji phantom4': 'quadcopter',
    'djimatrice': 'quadcopter',
    'dji_matrice': 'quadcopter',
    'dji matrice': 'quadcopter',
    'dji_matrice600': 'quadcopter',
    'dji matrice600': 'quadcopter',
    'quadcopter': 'quadcopter',
    'kraken': 'kraken',
    'opterra': 'opterra',
    'lrquad': 'lrquad',
    'unknown': 'unknown_drone',
    'unknown_drone': 'unknown_drone'
}

def classify_light_condition_from_path(path):
    """
    Classify light condition based on folder path
    Returns the light condition name if found in path
    """
    if 'bright_light' in path:
        return 'bright_light'
    elif 'medium_light' in path:
        return 'medium_light'
    elif 'low_light' in path:
        return 'low_light'
    else:
        return 'unknown_light'

def process_rgb_image(img):
    """Process RGB image (any necessary preprocessing)"""
    # Add any image preprocessing steps here if needed
    return img

def get_class_id_from_drone_type(drone_type):
    """Map drone type or alias to class ID for multi-class detection"""
    if not drone_type:
        return BASE_CLASS_MAPPING['unknown_drone']
    canonical_name = DRONE_TYPE_ALIASES.get(drone_type.lower(), drone_type.lower())
    return BASE_CLASS_MAPPING.get(canonical_name, BASE_CLASS_MAPPING['unknown_drone'])

def get_class_name_from_drone_type(drone_type):
    """Map drone type or alias to canonical class name"""
    if not drone_type:
        return 'unknown_drone'
    canonical_name = DRONE_TYPE_ALIASES.get(drone_type.lower(), drone_type.lower())
    if canonical_name in BASE_CLASS_MAPPING:
        return canonical_name
    return 'unknown_drone'

def process_mask_to_yolo(mask, agent_colors, img_shape, drone_type, class_override=None):
    """Process a mask image to YOLO format labels with multi-class support"""
    labels = []
    bboxes = []
    
    height, width = mask.shape[:2]
    
    # Get class ID for this drone type
    class_key = class_override if class_override else drone_type
    class_id = get_class_id_from_drone_type(class_key)
    class_name = get_class_name_from_drone_type(class_key)
    
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
                
                label = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                
                labels.append(label)
                bboxes.append((x_center, y_center, w_norm, h_norm, class_name))
    
    return labels, bboxes

def process_rgb_mask_pair(rgb_path, mask_path, agent_colors, output_dir, light_condition, drone_type, debug_visual_images=True):
    """Process a single rgb/mask directory pair"""
    no_bbox_count = 0
    total_count = 0
    
    # Get all RGB images
    rgb_files = [f for f in os.listdir(rgb_path) if f.endswith(('.png', '.jpg'))]
    total_count = len(rgb_files)
    
    print(f"\nProcessing {len(rgb_files)} RGB files in {rgb_path}")
    
    # Calculate how many images to debug (10% of total)
    debug_count = max(1, int(len(rgb_files) * 0.1))
    debug_indices = set(np.linspace(0, len(rgb_files)-1, debug_count, dtype=int))
    
    # Add progress bar
    for idx, rgb_file in enumerate(tqdm(rgb_files, desc=f"Processing {light_condition} images", unit="image")):
        # Get base name without extension and remove "_lit" suffix if present
        base_name = os.path.splitext(rgb_file)[0]
        if base_name.endswith('_lit'):
            base_name = base_name[:-4]  # Remove "_lit" suffix
        
        # Construct mask filename
        mask_file = f"{base_name}_object_mask.png"
        mask_path_full = os.path.join(mask_path, mask_file)
        
        try:
            rgb_img = cv2.imread(os.path.join(rgb_path, rgb_file))
            mask_img = cv2.imread(mask_path_full, cv2.IMREAD_UNCHANGED)
            
            if rgb_img is None or mask_img is None:
                print(f"\nWarning: Could not read images for {base_name}")
                continue
            
            # Process mask conversion if needed
            if mask_img is not None:
                if len(mask_img.shape) == 2:  # If grayscale
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
                elif mask_img.shape[2] == 4:  # If RGBA
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2BGR)
            
            # Process and save the RGB image
            processed_rgb = process_rgb_image(rgb_img)
            output_rgb_path = os.path.join(output_dir, 'images', rgb_file)
            try:
                cv2.imwrite(output_rgb_path, processed_rgb)
            except Exception as e:
                print(f"Error saving RGB file {output_rgb_path}: {e}")
                continue
            
            # Process and save the YOLO format label
            try:
                yolo_labels, bboxes = process_mask_to_yolo(mask_img, agent_colors, rgb_img.shape, drone_type)
                if not yolo_labels:  # If no labels were generated
                    no_bbox_count += 1
                with open(os.path.join(output_dir, 'labels', f"{base_name}_lit.txt"), 'w') as f:
                    for label in yolo_labels:
                        f.write(f"{label}\n")
            except Exception as e:
                print(f"Error processing mask to YOLO format: {e}")
                continue
            
            # Save debug visualization if enabled and this is one of the selected debug images
            if debug_visual_images and bboxes and idx in debug_indices:
                debug_img = processed_rgb.copy()
                height, width = debug_img.shape[:2]
                
                for bbox in bboxes:
                    x, y, w, h, class_name = bbox
                    # Convert normalized coordinates back to pixel coordinates
                    x1 = int((x - w/2) * width)
                    y1 = int((y - h/2) * height)
                    x2 = int((x + w/2) * width)
                    y2 = int((y + h/2) * height)
                    
                    # Draw rectangle and label with class name
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(debug_img, class_name, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imwrite(os.path.join(output_dir, 'debug-images', f"{base_name}_lit_debug.jpg"), debug_img)
        except Exception as e:
            print(f"Error processing files: {e}")
            continue
    
    return no_bbox_count, total_count

def find_rgb_mask_pairs(directory, base_input_path, base_output_path, debug_visual_images=True):
    """Recursively find directories containing both 'rgb' and 'mask' subdirectories"""
    results = []
    light_condition_stats = {
        'bright_light': 0,
        'medium_light': 0,
        'low_light': 0,
        'unknown_light': 0
    }
    drone_type_stats = {}
    
    # Check if current directory has both rgb and mask subdirectories
    rgb_path = os.path.join(directory, 'rgb')
    mask_path = os.path.join(directory, 'mask')
    
    if os.path.exists(rgb_path) and os.path.exists(mask_path):
        # This directory contains a valid rgb/mask pair
        rel_path = os.path.relpath(directory, base_input_path)
        output_dir = os.path.join(base_output_path, rel_path)
        
        # Extract drone type from path structure: base_input_path/drone_type/level/...
        path_parts = Path(rel_path).parts
        drone_type = path_parts[0] if len(path_parts) > 0 else 'unknown_drone'
        
        # Classify light condition based on path
        light_condition = classify_light_condition_from_path(directory)
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        if debug_visual_images:
            os.makedirs(os.path.join(output_dir, 'debug-images'), exist_ok=True)
        
        # Check for agent_color_info.json
        agent_info_path = os.path.join(directory, 'agent_color_info.json')
        if not os.path.exists(agent_info_path):
            # All masks are yellow (same physical drone DJIS900), but subfolder indicates the class
            agent_colors = {
                "drone": [0, 255, 255]  # RGB(255,255,0) yellow in BGR format (all masks are yellow)
            }
            print(f"No agent_color_info.json found in {directory}, using yellow color (all masks are yellow). Subfolder '{drone_type}' indicates target class.")
        else:
            # Load agent color information
            with open(agent_info_path, 'r') as f:
                agent_colors = json.load(f)
        
        # Process this pair
        no_bbox_count, total_count = process_rgb_mask_pair(rgb_path, mask_path, agent_colors, output_dir, light_condition, drone_type, debug_visual_images)
        results.append((directory, no_bbox_count, total_count, light_condition, drone_type))
        
        # Update light condition and drone type stats
        light_condition_stats[light_condition] += total_count
        if drone_type not in drone_type_stats:
            drone_type_stats[drone_type] = 0
        drone_type_stats[drone_type] += total_count
    else:
        # Check subdirectories
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # Recursively search this subdirectory
                sub_results, sub_light_stats, sub_drone_stats = find_rgb_mask_pairs(item_path, base_input_path, base_output_path, debug_visual_images)
                results.extend(sub_results)
                # Merge light condition stats
                for condition, count in sub_light_stats.items():
                    light_condition_stats[condition] += count
                # Merge drone type stats
                for drone_type, count in sub_drone_stats.items():
                    if drone_type not in drone_type_stats:
                        drone_type_stats[drone_type] = 0
                    drone_type_stats[drone_type] += count
    
    return results, light_condition_stats, drone_type_stats

def process_dataset(base_input_path, base_output_path, debug_visual_images=True):
    """Process the BO dataset with the existing folder structure"""
    
    print(f"Processing dataset from: {base_input_path}")
    print(f"Output will be saved to: {base_output_path}")
    
    # Initialize global counters
    global_total_images = 0
    global_no_bbox_images = 0
    global_light_condition_stats = {
        'bright_light': 0,
        'medium_light': 0,
        'low_light': 0,
        'unknown_light': 0
    }
    global_drone_type_stats = {}
    
    # Find all rgb/mask pairs recursively
    print("\nSearching for RGB/mask pairs...")
    results, light_stats, drone_stats = find_rgb_mask_pairs(base_input_path, base_input_path, base_output_path, debug_visual_images)
    
    # Process results and collect statistics
    for rgb_mask_dir, bbox_count, total_count, light_condition, drone_type in results:
        global_total_images += total_count
        global_no_bbox_images += bbox_count
        
    # Merge light condition stats
    for condition, count in light_stats.items():
        global_light_condition_stats[condition] += count
    
    # Merge drone type stats
    for drone_type, count in drone_stats.items():
        if drone_type not in global_drone_type_stats:
            global_drone_type_stats[drone_type] = 0
        global_drone_type_stats[drone_type] += count
    
    # Print processing statistics
    print(f"\n=== PROCESSING STATISTICS ===")
    print(f"Total images processed: {global_total_images}")
    print(f"Images with no bounding boxes: {global_no_bbox_images}")
    if global_total_images > 0:
        print(f"Percentage with no bounding boxes: {global_no_bbox_images/global_total_images*100:.2f}%")
    
    # Generate and save light condition report
    report_path = os.path.join(base_output_path, "light_condition_report.txt")
    total_light_images = sum(global_light_condition_stats.values())
    
    with open(report_path, 'w') as f:
        f.write("=== BAYESIAN OPTIMIZATION DATASET LIGHT CONDITION REPORT ===\n\n")
        f.write(f"Processing Date: {Path().absolute()}\n")
        f.write(f"Input Directory: {base_input_path}\n")
        f.write(f"Output Directory: {base_output_path}\n\n")
        
        f.write("=== LIGHT CONDITION BREAKDOWN ===\n")
        for condition in ['bright_light', 'medium_light', 'low_light', 'unknown_light']:
            count = global_light_condition_stats[condition]
            percentage = (count / total_light_images * 100) if total_light_images > 0 else 0
            f.write(f"{condition.upper()}: {count} images ({percentage:.2f}%)\n")
        
        f.write(f"\nTOTAL RGB IMAGES: {total_light_images}\n")
        
        f.write(f"\n=== PROCESSING STATISTICS ===\n")
        f.write(f"Total images processed: {global_total_images}\n")
        f.write(f"Images with no bounding boxes: {global_no_bbox_images}\n")
        if global_total_images > 0:
            f.write(f"Percentage with no bounding boxes: {global_no_bbox_images/global_total_images*100:.2f}%\n")
        
        f.write(f"\n=== DRONE TYPE BREAKDOWN ===\n")
        for drone_type, count in global_drone_type_stats.items():
            percentage = (count / total_light_images * 100) if total_light_images > 0 else 0
            f.write(f"{drone_type.upper()}: {count} images ({percentage:.2f}%)\n")
        
        f.write(f"\n=== DETAILED BREAKDOWN BY ENVIRONMENT ===\n")
        # Group results by environment level (drone_type/level structure)
        environment_stats = {}
        for rgb_mask_dir, bbox_count, total_count, light_condition, drone_type in results:
            # Extract environment name from path: drone_type/level/...
            rel_path = os.path.relpath(rgb_mask_dir, base_input_path)
            path_parts = Path(rel_path).parts
            if len(path_parts) > 1:
                env_name = f"{path_parts[0]}/{path_parts[1]}"  # drone_type/level
                if env_name not in environment_stats:
                    environment_stats[env_name] = {
                        'bright_light': 0,
                        'medium_light': 0,
                        'low_light': 0,
                        'unknown_light': 0
                    }
                environment_stats[env_name][light_condition] += total_count
        
        for env_name, stats in environment_stats.items():
            f.write(f"\nEnvironment: {env_name}\n")
            env_total = sum(stats.values())
            for condition in ['bright_light', 'medium_light', 'low_light', 'unknown_light']:
                count = stats[condition]
                percentage = (count / env_total * 100) if env_total > 0 else 0
                f.write(f"  {condition}: {count} images ({percentage:.2f}%)\n")
    
    # Print report summary to console
    print(f"\n=== LIGHT CONDITION BREAKDOWN ===")
    for condition in ['bright_light', 'medium_light', 'low_light', 'unknown_light']:
        count = global_light_condition_stats[condition]
        percentage = (count / total_light_images * 100) if total_light_images > 0 else 0
        print(f"{condition.upper()}: {count} images ({percentage:.2f}%)")
    
    print(f"\n=== DRONE TYPE BREAKDOWN ===")
    for drone_type, count in global_drone_type_stats.items():
        percentage = (count / total_light_images * 100) if total_light_images > 0 else 0
        print(f"{drone_type.upper()}: {count} images ({percentage:.2f}%)")
    
    print(f"\nTOTAL RGB IMAGES: {total_light_images}")
    print(f"\nDetailed report saved to: {report_path}")
    
    return global_light_condition_stats, global_drone_type_stats

def prepare_yolo_dataset(base_output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, drone_types=None, delete_empty=True):
    """
    Prepare YOLO-compatible dataset structure with train/val/test split
    
    Args:
        base_output_path: Path to the processed dataset
        train_ratio: Ratio for training set (default 0.8 = 80%)
        val_ratio: Ratio for validation set (default 0.1 = 10%)
        test_ratio: Ratio for test set (default 0.1 = 10%)
        drone_types: List of drone types detected
        delete_empty: If True, exclude images with empty label files (default True)
    """
    print(f"\n=== PREPARING YOLO DATASET ===")
    print(f"Train ratio: {train_ratio*100}%")
    print(f"Validation ratio: {val_ratio*100}%")
    print(f"Test ratio: {test_ratio*100}%")
    print(f"Delete empty labels: {delete_empty}")
    
    # Create YOLO dataset directory structure
    yolo_dir = os.path.join(base_output_path, "yolo_dataset")
    train_images_dir = os.path.join(yolo_dir, "train", "images")
    train_labels_dir = os.path.join(yolo_dir, "train", "labels")
    val_images_dir = os.path.join(yolo_dir, "val", "images")
    val_labels_dir = os.path.join(yolo_dir, "val", "labels")
    test_images_dir = os.path.join(yolo_dir, "test", "images")
    test_labels_dir = os.path.join(yolo_dir, "test", "labels")
    
    # Create directories
    for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Collect all image-label pairs from processed dataset
    image_label_pairs = []
    
    # Walk through the processed dataset to find all images and labels
    for root, dirs, files in os.walk(base_output_path):
        # Skip the yolo_dataset directory itself
        if 'yolo_dataset' in root:
            continue
            
        # Check if this directory contains both images and labels
        if 'images' in dirs and 'labels' in dirs:
            images_path = os.path.join(root, 'images')
            labels_path = os.path.join(root, 'labels')
            
            # Get all image files
            image_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                # Determine corresponding label file name
                base_name = os.path.splitext(img_file)[0]
                
                # Handle different naming conventions
                label_file = f"{base_name}.txt"
                if not os.path.exists(os.path.join(labels_path, label_file)):
                    # Try with _lit suffix removed
                    if base_name.endswith('_lit'):
                        label_file = f"{base_name}.txt"
                    else:
                        label_file = f"{base_name}_lit.txt"
                
                label_path = os.path.join(labels_path, label_file)
                img_path = os.path.join(images_path, img_file)
                
                # Check if both files exist
                if os.path.exists(label_path) and os.path.exists(img_path):
                    # Check if we should include empty labels or not
                    include_pair = True
                    if delete_empty and os.path.getsize(label_path) == 0:
                        include_pair = False
                    
                    if include_pair:
                        # Create unique filename to avoid conflicts
                        rel_path = os.path.relpath(root, base_output_path)
                        unique_prefix = rel_path.replace(os.sep, '_').replace(' ', '_')
                        unique_img_name = f"{unique_prefix}_{img_file}"
                        unique_label_name = f"{unique_prefix}_{os.path.splitext(img_file)[0]}.txt"
                        
                        image_label_pairs.append({
                            'img_src': img_path,
                            'label_src': label_path,
                            'img_name': unique_img_name,
                            'label_name': unique_label_name
                        })
    
    print(f"Found {len(image_label_pairs)} valid image-label pairs")
    
    if len(image_label_pairs) == 0:
        print("No valid image-label pairs found!")
        return
    
    # Shuffle and split the dataset
    random.shuffle(image_label_pairs)
    train_count = int(len(image_label_pairs) * train_ratio)
    val_count = int(len(image_label_pairs) * val_ratio)
    
    train_pairs = image_label_pairs[:train_count]
    val_pairs = image_label_pairs[train_count:train_count + val_count]
    test_pairs = image_label_pairs[train_count + val_count:]
    
    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Validation set: {len(val_pairs)} pairs")
    print(f"Test set: {len(test_pairs)} pairs")
    
    # Copy files to YOLO structure
    def copy_pairs(pairs, img_dir, label_dir, split_name):
        print(f"Copying {split_name} files...")
        for pair in tqdm(pairs, desc=f"Copying {split_name}"):
            try:
                # Copy image
                dest_img_path = os.path.join(img_dir, pair['img_name'])
                shutil.copy2(pair['img_src'], dest_img_path)
                
                # Copy label
                dest_label_path = os.path.join(label_dir, pair['label_name'])
                shutil.copy2(pair['label_src'], dest_label_path)
                
            except Exception as e:
                print(f"Error copying {pair['img_name']}: {e}")
    
    # Copy train, validation, and test sets
    copy_pairs(train_pairs, train_images_dir, train_labels_dir, "train")
    copy_pairs(val_pairs, val_images_dir, val_labels_dir, "validation")
    copy_pairs(test_pairs, test_images_dir, test_labels_dir, "test")
    
    # Create dataset.yaml file with detected drone types
    create_dataset_yaml(yolo_dir, train_images_dir, val_images_dir, test_images_dir, drone_types)
    
    print(f"\nYOLO dataset created successfully at: {yolo_dir}")
    print(f"Train images: {len(os.listdir(train_images_dir))}")
    print(f"Train labels: {len(os.listdir(train_labels_dir))}")
    print(f"Val images: {len(os.listdir(val_images_dir))}")
    print(f"Val labels: {len(os.listdir(val_labels_dir))}")
    print(f"Test images: {len(os.listdir(test_images_dir))}")
    print(f"Test labels: {len(os.listdir(test_labels_dir))}")
    
    return yolo_dir

def sanitize_for_filename(value):
    """Sanitize a string for safe filename usage."""
    if value is None:
        return "data"
    sanitized = value.replace("\\", "_").replace("/", "_").replace(":", "_")
    sanitized = sanitized.replace(" ", "_")
    sanitized = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in sanitized)
    sanitized = sanitized.strip("_")
    return sanitized if sanitized else "data"


def load_agent_colors(agent_info_path):
    """Load agent color information, defaulting to yellow if unavailable."""
    if agent_info_path and os.path.exists(agent_info_path):
        try:
            with open(agent_info_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return data
        except Exception as exc:
            print(f"Warning: Failed to load agent_color_info.json from {agent_info_path}: {exc}")
    return {"drone": [0, 255, 255]}  # Default yellow in BGR


def load_source_config(config_path):
    """
    Load multi-source dataset configuration from YAML or JSON file.
    The configuration must resolve to a list of mappings with the schema:

        - class_name: <canonical class name>
          sample_ratio: <float, optional>
          paths:
            - path: <dataset directory>
              default_drone_type: <optional override>
              prefix: <optional unique prefix override>

    Returns a normalized list compatible with create_yolo_dataset_from_source_configs.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ('.yaml', '.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to parse configuration file '{config_path}': {exc}") from exc

    if isinstance(data, dict):
        if 'sources' in data:
            data = data['sources']
        elif 'classes' in data:
            data = data['classes']
        else:
            # Allow dict of named entries -> convert to list
            if all(isinstance(value, dict) for value in data.values()):
                data = list(data.values())
            else:
                data = [data]

    if not isinstance(data, list):
        raise ValueError("Source configuration must be a list of class definitions.")

    normalized = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Configuration entry #{idx} must be a mapping, got {type(entry).__name__}")

        class_name = entry.get('class_name')
        if not class_name:
            raise ValueError(f"Configuration entry #{idx} is missing required key 'class_name'.")

        paths = entry.get('paths')
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"Configuration for class '{class_name}' must include a non-empty 'paths' list.")

        sample_ratio = entry.get('sample_ratio', 1.0)
        try:
            sample_ratio = float(sample_ratio)
        except (TypeError, ValueError):
            raise ValueError(f"'sample_ratio' for class '{class_name}' must be numeric.") from None

        normalized_entry = {
            key: value
            for key, value in entry.items()
            if key not in ('class_name', 'sample_ratio', 'paths')
        }
        normalized_entry['class_name'] = class_name
        normalized_entry['sample_ratio'] = sample_ratio
        normalized_entry['paths'] = []

        for path_idx, path_cfg in enumerate(paths):
            if isinstance(path_cfg, str):
                normalized_entry['paths'].append({'path': path_cfg})
                continue
            if not isinstance(path_cfg, dict):
                raise ValueError(
                    f"Path definition #{path_idx} for class '{class_name}' must be a string or mapping."
                )
            if 'path' not in path_cfg or not path_cfg['path']:
                raise ValueError(
                    f"Path definition #{path_idx} for class '{class_name}' is missing required key 'path'."
                )
            normalized_entry['paths'].append(dict(path_cfg))

        normalized.append(normalized_entry)

    return normalized


def collect_metadata_from_directory(base_input_path, class_name=None, default_source_drone_type=None, prefix=None):
    """
    Collect lightweight metadata entries for each rgb/mask pair inside base_input_path.

    Args:
        base_input_path: Dataset directory to search (must contain rgb and mask subfolders)
        class_name: Optional canonical class override; if None, inferred from source drone type
        default_source_drone_type: Optional drone type to use when it cannot be inferred from path
        prefix: Optional prefix to use for unique filenames
    """
    metadata = []
    base_input_path = os.path.abspath(base_input_path)

    if not os.path.exists(base_input_path):
        print(f"Warning: Input path {base_input_path} does not exist. Skipping.")
        return metadata

    base_prefix = sanitize_for_filename(prefix) if prefix else sanitize_for_filename(Path(base_input_path).name)
    print(f"\nScanning dataset for file paths in: {base_input_path}")

    for root, dirs, files in os.walk(base_input_path):
        rgb_path = os.path.join(root, 'rgb')
        mask_path = os.path.join(root, 'mask')

        if not (os.path.exists(rgb_path) and os.path.exists(mask_path)):
            continue

        rel_path = os.path.relpath(root, base_input_path)
        path_parts = Path(rel_path).parts if rel_path not in ('.', '') else ()
        source_drone_type = default_source_drone_type or (path_parts[0] if len(path_parts) > 0 else (class_name or 'unknown_drone'))
        if not source_drone_type:
            source_drone_type = 'unknown_drone'
        canonical_class = class_name or get_class_name_from_drone_type(source_drone_type)

        agent_info_path = os.path.join(root, 'agent_color_info.json')
        rgb_files = [f for f in os.listdir(rgb_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Found {len(rgb_files)} images in {rel_path}")

        rel_prefix = sanitize_for_filename(rel_path) if rel_path not in ('.', '') else ''
        if rel_prefix:
            unique_prefix = sanitize_for_filename(f"{canonical_class}_{base_prefix}_{rel_prefix}")
        else:
            unique_prefix = sanitize_for_filename(f"{canonical_class}_{base_prefix}")

        for rgb_file in rgb_files:
            base_name = os.path.splitext(rgb_file)[0]
            base_name_no_suffix = base_name[:-4] if base_name.endswith('_lit') else base_name
            mask_filename = f"{base_name_no_suffix}_object_mask.png"
            mask_path_full = os.path.join(mask_path, mask_filename)
            rgb_path_full = os.path.join(rgb_path, rgb_file)

            if not os.path.exists(mask_path_full):
                continue

            metadata.append({
                'rgb_path': rgb_path_full,
                'mask_path': mask_path_full,
                'agent_info_path': agent_info_path if os.path.exists(agent_info_path) else None,
                'class_name': canonical_class,
                'source_drone_type': source_drone_type,
                'rgb_filename': rgb_file,
                'mask_filename': mask_filename,
                'label_stem': os.path.splitext(rgb_file)[0],
                'unique_prefix': unique_prefix,
                'base_input_path': base_input_path,
                'relative_dir': rel_path if rel_path not in ('.', '') else ''
            })

    print(f"Collected {len(metadata)} metadata entries from {base_input_path}")
    return metadata


def sample_metadata_by_group(metadata_list, sample_ratio, rng=None, group_key='source_drone_type'):
    """
    Sample metadata entries per group (e.g., per drone type).

    Args:
        metadata_list: List of metadata dictionaries
        sample_ratio: Fraction to keep (0-1]. If >=1, returns original list
        rng: Optional random.Random instance for deterministic sampling
        group_key: Key used to group metadata before sampling
    """
    if sample_ratio <= 0.0:
        return []
    if sample_ratio >= 1.0:
        return list(metadata_list)

    sampler = rng if rng is not None else random
    grouped = defaultdict(list)
    for item in metadata_list:
        grouped[item.get(group_key, 'unknown')].append(item)

    sampled = []
    for group, items in grouped.items():
        if not items:
            continue
        target_count = max(1, int(math.ceil(len(items) * sample_ratio)))
        target_count = min(len(items), target_count)
        if target_count >= len(items):
            sampled.extend(items)
        else:
            if isinstance(sampler, random.Random):
                sampled.extend(sampler.sample(items, target_count))
            else:
                sampled.extend(random.sample(items, target_count))
        print(f"Sampling group '{group}': keeping {target_count} of {len(items)} entries")

    return sampled


def process_metadata_into_yolo(metadata_list, base_output_path, train_ratio, val_ratio, test_ratio,
                               delete_empty, debug_visual_images, seed=None):
    """
    Process collected metadata to build YOLO dataset splits.
    Returns tuple of (yolo_dir, class_stats dictionary).
    """
    yolo_dir = os.path.join(base_output_path, "yolo_dataset")
    train_images_dir = os.path.join(yolo_dir, "train", "images")
    train_labels_dir = os.path.join(yolo_dir, "train", "labels")
    val_images_dir = os.path.join(yolo_dir, "val", "images")
    val_labels_dir = os.path.join(yolo_dir, "val", "labels")
    test_images_dir = os.path.join(yolo_dir, "test", "images")
    test_labels_dir = os.path.join(yolo_dir, "test", "labels")

    for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir]:
        os.makedirs(directory, exist_ok=True)

    debug_dir = None
    if debug_visual_images:
        debug_dir = os.path.join(yolo_dir, "debug-images")
        os.makedirs(debug_dir, exist_ok=True)

    if not metadata_list:
        print("No metadata entries available for YOLO dataset creation.")
        return yolo_dir, {}

    metadata_copy = list(metadata_list)
    if seed is not None:
        shuffle_rng = random.Random(seed)
        shuffle_rng.shuffle(metadata_copy)
    else:
        random.shuffle(metadata_copy)

    total_pairs = len(metadata_copy)
    train_count = int(total_pairs * train_ratio)
    val_count = int(total_pairs * val_ratio)

    train_metadata = metadata_copy[:train_count]
    val_metadata = metadata_copy[train_count:train_count + val_count]
    test_metadata = metadata_copy[train_count + val_count:]

    print(f"Planned splits - Train: {len(train_metadata)}, Val: {len(val_metadata)}, Test: {len(test_metadata)}")

    class_stats = defaultdict(int)

    def process_and_save_split(metadata_subset, img_dir, label_dir, split_name):
        if not metadata_subset:
            print(f"\nNo files to process for {split_name} split.")
            return 0, 0, {}

        print(f"\nProcessing and saving {split_name} files...")
        debug_indices = set()
        if debug_visual_images and debug_dir:
            debug_count = max(1, int(len(metadata_subset) * 0.1))
            debug_count = min(debug_count, len(metadata_subset))
            if debug_count > 0:
                debug_indices = set(np.linspace(0, len(metadata_subset) - 1, debug_count, dtype=int))

        saved_count = 0
        skipped_count = 0
        split_class_counts = defaultdict(int)

        for idx, metadata in enumerate(tqdm(metadata_subset, desc=f"Processing {split_name}")):
            try:
                agent_colors = load_agent_colors(metadata.get('agent_info_path'))
                rgb_img = cv2.imread(metadata['rgb_path'])
                mask_img = cv2.imread(metadata['mask_path'], cv2.IMREAD_UNCHANGED)

                if rgb_img is None or mask_img is None:
                    print(f"Warning: Unable to read RGB or mask for {metadata['rgb_path']}")
                    continue

                if len(mask_img.shape) == 2:
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
                elif mask_img.shape[2] == 4:
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2BGR)

                yolo_labels, bboxes = process_mask_to_yolo(
                    mask_img,
                    agent_colors,
                    rgb_img.shape,
                    metadata['source_drone_type'],
                    class_override=metadata['class_name']
                )

                has_labels = len(yolo_labels) > 0
                if delete_empty and not has_labels:
                    skipped_count += 1
                    continue

                img_name = f"{metadata['unique_prefix']}_{metadata['rgb_filename']}"
                label_name = f"{metadata['unique_prefix']}_{metadata['label_stem']}.txt"

                img_path = os.path.join(img_dir, img_name)
                label_path = os.path.join(label_dir, label_name)

                try:
                    cv2.imwrite(img_path, rgb_img)
                except Exception as exc:
                    print(f"Error saving RGB image {img_path}: {exc}")
                    continue

                try:
                    with open(label_path, 'w') as f:
                        for label in yolo_labels:
                            f.write(f"{label}\n")
                except Exception as exc:
                    print(f"Error writing label file {label_path}: {exc}")
                    continue

                if debug_visual_images and debug_dir and bboxes and idx in debug_indices:
                    debug_img = rgb_img.copy()
                    height, width = debug_img.shape[:2]
                    for bbox in bboxes:
                        x, y, w, h, class_name = bbox
                        x1 = int((x - w/2) * width)
                        y1 = int((y - h/2) * height)
                        x2 = int((x + w/2) * width)
                        y2 = int((y + h/2) * height)
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(debug_img, class_name, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    debug_filename = f"{split_name}_{metadata['unique_prefix']}_{metadata['label_stem']}_debug.jpg"
                    try:
                        cv2.imwrite(os.path.join(debug_dir, debug_filename), debug_img)
                    except Exception as exc:
                        print(f"Error saving debug image {debug_filename}: {exc}")

                split_class_counts[metadata['class_name']] += 1
                saved_count += 1

                del rgb_img, mask_img
                if 'debug_img' in locals():
                    del debug_img

            except Exception as exc:
                print(f"Error processing {metadata['rgb_path']}: {exc}")
                continue

        print(f"{split_name} - Saved: {saved_count}, Skipped (empty): {skipped_count}")
        return saved_count, skipped_count, split_class_counts

    train_saved, train_skipped, train_stats = process_and_save_split(train_metadata, train_images_dir, train_labels_dir, "train")
    val_saved, val_skipped, val_stats = process_and_save_split(val_metadata, val_images_dir, val_labels_dir, "val")
    test_saved, test_skipped, test_stats = process_and_save_split(test_metadata, test_images_dir, test_labels_dir, "test")

    for stats in [train_stats, val_stats, test_stats]:
        for class_name, count in stats.items():
            class_stats[class_name] += count

    detected_classes = list(class_stats.keys())
    create_dataset_yaml(yolo_dir, train_images_dir, val_images_dir, test_images_dir, detected_classes)

    total_processed = train_saved + val_saved + test_saved
    total_skipped = train_skipped + val_skipped + test_skipped

    print(f"\nYOLO dataset created successfully at: {yolo_dir}")
    print(f"Train images: {train_saved}")
    print(f"Val images: {val_saved}")
    print(f"Test images: {test_saved}")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (empty labels): {total_skipped}")

    if class_stats:
        print(f"\n=== CLASS BREAKDOWN ===")
        grand_total = sum(class_stats.values())
        for class_name, count in class_stats.items():
            percentage = (count / grand_total * 100) if grand_total > 0 else 0
            print(f"{class_name.upper()}: {count} images ({percentage:.2f}%)")

    return yolo_dir, class_stats


def create_yolo_dataset_direct(base_input_path, base_output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                               delete_empty=True, debug_visual_images=True, seed=None):
    """
    Create YOLO dataset directly from a single source dataset.
    """
    print(f"\n=== CREATING YOLO DATASET DIRECTLY (MEMORY-EFFICIENT) ===")
    print(f"Source dataset: {base_input_path}")
    print(f"Train ratio: {train_ratio*100}%")
    print(f"Validation ratio: {val_ratio*100}%")
    print(f"Test ratio: {test_ratio*100}%")
    print(f"Delete empty labels: {delete_empty}")

    metadata = collect_metadata_from_directory(base_input_path)
    if not metadata:
        print("No valid image-label pairs found!")
        return os.path.join(base_output_path, "yolo_dataset")

    yolo_dir, _ = process_metadata_into_yolo(
        metadata,
        base_output_path,
        train_ratio,
        val_ratio,
        test_ratio,
        delete_empty,
        debug_visual_images,
        seed=seed
    )

    return yolo_dir


def create_yolo_dataset_from_source_configs(source_configs, base_output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                                            delete_empty=True, debug_visual_images=True, seed=None):
    """
    Create YOLO dataset from a list of source configuration dictionaries.

    Each source configuration should include:
        {
            "class_name": "quadcopter",
            "sample_ratio": 0.3,
            "paths": [
                {"path": "...", "default_drone_type": "DJIS900", "prefix": "optional_prefix"}
            ]
        }
    """
    all_metadata = []

    for idx, source in enumerate(source_configs):
        class_name = source.get('class_name')
        if not class_name:
            print("Warning: Encountered source configuration without 'class_name'. Skipping.")
            continue

        sample_ratio = float(source.get('sample_ratio', 1.0))
        paths = source.get('paths', [])

        if not paths:
            print(f"Warning: No paths provided for class '{class_name}'. Skipping.")
            continue

        class_metadata = []
        for path_cfg in paths:
            path = path_cfg.get('path')
            if not path:
                continue
            default_drone_type = path_cfg.get('default_drone_type')
            prefix = path_cfg.get('prefix')
            class_metadata.extend(
                collect_metadata_from_directory(
                    path,
                    class_name=class_name,
                    default_source_drone_type=default_drone_type,
                    prefix=prefix
                )
            )

        if not class_metadata:
            print(f"Warning: No data collected for class '{class_name}'.")
            continue

        sample_ratio = max(0.0, min(1.0, sample_ratio))
        if sample_ratio <= 0.0:
            print(f"Sample ratio for class '{class_name}' is 0. Skipping this class.")
            continue

        if sample_ratio < 1.0:
            sample_seed = (seed + idx) if seed is not None else None
            rng = random.Random(sample_seed) if sample_seed is not None else None
            class_metadata = sample_metadata_by_group(class_metadata, sample_ratio, rng=rng)
            print(f"After sampling, {len(class_metadata)} items remain for class '{class_name}'.")
        else:
            print(f"Using all {len(class_metadata)} items for class '{class_name}'.")

        all_metadata.extend(class_metadata)

    if not all_metadata:
        print("No metadata collected from custom source configuration.")
        return os.path.join(base_output_path, "yolo_dataset")

    yolo_dir, _ = process_metadata_into_yolo(
        all_metadata,
        base_output_path,
        train_ratio,
        val_ratio,
        test_ratio,
        delete_empty,
        debug_visual_images,
        seed=seed
    )

    return yolo_dir

def create_dataset_yaml(yolo_dir, train_images_dir, val_images_dir, test_images_dir, drone_types=None):
    """Create the dataset.yaml file for YOLO training with multi-class support"""
    
    # Use forward slashes for cross-platform compatibility
    train_path = train_images_dir.replace("\\", "/")
    val_path = val_images_dir.replace("\\", "/")
    test_path = test_images_dir.replace("\\", "/")
    
    # Create class names list based on detected drone types
    if drone_types is None:
        drone_types = ['djis900', 'kraken', 'lrquad', 'opterra', 'unknown_drone']
    
    # Ensure consistent ordering with get_class_id_from_drone_type function
    class_names = []
    class_mapping = {}
    
    for drone_type in sorted(drone_types):
        class_id = get_class_id_from_drone_type(drone_type)
        class_name = get_class_name_from_drone_type(drone_type)
        class_mapping[class_id] = class_name
    
    # Create ordered list of class names
    max_class_id = max(class_mapping.keys())
    class_names = [''] * (max_class_id + 1)
    for class_id, class_name in class_mapping.items():
        class_names[class_id] = class_name
    
    # Remove empty entries
    class_names = [name for name in class_names if name]
    
    dataset_yaml = {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(class_names),  # Number of classes
        'names': class_names  # Class names
    }
    
    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"dataset.yaml created at: {yaml_path}")
    print(f"Classes: {class_names}")
    
    # Also create a more detailed yaml with metadata
    detailed_yaml = {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(class_names),
        'names': class_names,
        'description': 'SIDG-ATRID Multi-Class Drone Detection Dataset',
        'version': '2.0',
        'date_created': str(Path().absolute()),
        'classes': class_mapping,
        'drone_types_detected': list(drone_types)
    }
    
    detailed_yaml_path = os.path.join(yolo_dir, "dataset_detailed.yaml")
    with open(detailed_yaml_path, 'w') as f:
        yaml.dump(detailed_yaml, f, default_flow_style=False)
    
    print(f"Detailed dataset.yaml created at: {detailed_yaml_path}")

def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(description='Bayesian Optimization Dataset Post-Processor')
    parser.add_argument('--input', '-i', default=r"F:\SIDG-ATRID-Dataset\Train_set\sidg-atrid-kraken-dataset-v7",
                      help='Input directory path (default: F:\\SIDG-ATRID-Dataset\\flight_demo_data)')
    parser.add_argument('--output', '-o', default=r"F:\SIDG-ATRID-Dataset\Train_set\sidg-atrid-dataset-v8-yolo_processed",
                      help='Output directory path (default: F:\\SIDG-ATRID-Dataset\\flight_demo_data_processed_2)')
    parser.add_argument('--debug', action='store_true', default=True,
                      help='Enable debug visual images (default: True)')
    parser.add_argument('--yolo', action='store_true',
                      help='Prepare YOLO dataset structure with train/val split')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Training set ratio (default: 0.8 = 80%%)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                      help='Validation set ratio (default: 0.1 = 10%%)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                      help='Test set ratio (default: 0.1 = 10%%)')
    parser.add_argument('--keep-empty', action='store_true', default=False,
                      help='Keep image-label pairs with empty labels (includes images without drones). Default: delete empty labels.')
    parser.add_argument('--source-config', type=str,
                      help='Path to YAML or JSON file describing multi-source dataset configuration for YOLO mode.')
    parser.add_argument('--seed', type=int, default=None,
                      help='Optional random seed for shuffling and sampling operations.')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error: Train ratio ({args.train_ratio}) + Val ratio ({args.val_ratio}) + Test ratio ({args.test_ratio}) = {total_ratio:.3f}, must equal 1.0")
        return
    
    # Handle empty label deletion logic - default to True (backward compatibility)
    delete_empty = not args.keep_empty  # Simple: if keep_empty is True, delete_empty is False
    
    base_input_path = args.input
    base_output_path = args.output
    debug_visual_images = args.debug
    
    print("=== BAYESIAN OPTIMIZATION DATASET POST-PROCESSOR ===")
    print(f"Input directory: {base_input_path}")
    print(f"Output directory: {base_output_path}")
    print(f"Debug images enabled: {debug_visual_images}")
    print(f"Delete empty labels: {delete_empty}")
    if args.yolo:
        print(f"YOLO preparation enabled: {args.train_ratio*100}% train, {args.val_ratio*100}% val, {args.test_ratio*100}% test")
    if args.source_config:
        print(f"Using source configuration file: {args.source_config}")
    
    # Check if input directory exists
    if not args.source_config and not os.path.exists(base_input_path):
        print(f"Error: Input directory {base_input_path} does not exist!")
        return
    elif args.source_config and not os.path.exists(base_input_path):
        print(f"Warning: Input directory {base_input_path} does not exist and will be ignored because --source-config is provided.")
    
    # Create output directory
    os.makedirs(base_output_path, exist_ok=True)
    
    yolo_dir = None

    # Choose processing mode based on --yolo flag
    if args.yolo:
        if args.source_config:
            print("YOLO mode: Creating YOLO dataset from source configuration")
            try:
                source_configs = load_source_config(args.source_config)
            except Exception as exc:
                print(f"Error loading source configuration: {exc}")
                return

            print(f"Loaded {len(source_configs)} source class definitions from {args.source_config}")
            yolo_dir = create_yolo_dataset_from_source_configs(
                source_configs,
                base_output_path,
                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
                delete_empty,
                debug_visual_images,
                seed=args.seed
            )
        else:
            print("YOLO mode: Creating YOLO dataset directly from source data")
            yolo_dir = create_yolo_dataset_direct(
                base_input_path,
                base_output_path,
                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
                delete_empty,
                debug_visual_images,
                seed=args.seed
            )
    else:
        if args.source_config:
            print("Error: --source-config can only be used with --yolo mode.")
            return
        print("Processing mode: Creating processed dataset structure")
        # Process the dataset normally
        light_stats, drone_stats = process_dataset(base_input_path, base_output_path, debug_visual_images)
    
    print("\n=== PROCESSING COMPLETE ===")
    print(f"All processed data saved to: {base_output_path}")
    if args.yolo:
        print(f"YOLO dataset ready at: {yolo_dir}")

if __name__ == "__main__":
    main()
