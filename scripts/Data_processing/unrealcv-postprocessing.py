import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# Define paths
base_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput"
dataset_dir = os.path.join(base_dir, "dataset")
agent_color_info_path = os.path.join(base_dir, "clearsky", "agent_color_info.json")

# Create dataset subfolders if not present
folders = [
    "images/train", "images/val", "images/test",
    "labels/train", "labels/val", "labels/test"
]
for folder in folders:
    path = os.path.join(dataset_dir, folder)
    os.makedirs(path, exist_ok=True)

# Load agent color information
with open(agent_color_info_path, 'r') as f:
    agent_color_info = json.load(f)

# Map RGB colors to class IDs
color_to_class_id = {tuple((v['R'], v['G'], v['B'])): i for i, v in enumerate(agent_color_info.values())}

# Get all mask and corresponding RGB images
mask_dir = os.path.join(base_dir, "clearsky", "mask")
rgb_dir = os.path.join(base_dir, "clearsky", "rgb")
mask_files = [f for f in os.listdir(mask_dir) if f.endswith("_object_mask.png")]

# Process each mask image to create YOLO annotation files
for mask_file in mask_files:
    mask_path = os.path.join(mask_dir, mask_file)
    rgb_filename = mask_file.replace("_object_mask.png", "_lit.png")
    rgb_path = os.path.join(rgb_dir, rgb_filename)

    if not os.path.exists(rgb_path):
        print(f"Skipping {rgb_filename} as it does not exist in the RGB directory.")
        continue

    # Debug print to show the file being processed
    print(f"Processing mask: {mask_file} with corresponding RGB: {rgb_filename}")

    # Load mask image
    mask_img = cv2.imread(mask_path)

    # Initialize annotation data
    height, width, _ = mask_img.shape
    annotations = []

    # Detect unique colors in mask
    unique_colors = np.unique(mask_img.reshape(-1, mask_img.shape[2]), axis=0)

    for color in unique_colors:
        color_tuple = tuple(color[:3])
        if color_tuple in color_to_class_id:
            class_id = color_to_class_id[color_tuple]

            # Create a binary mask for the specific color
            mask = cv2.inRange(mask_img, color, color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = (x + w / 2) / width
                center_y = (y + h / 2) / height
                norm_w = w / width
                norm_h = h / height

                # Append annotation: class_id, center_x, center_y, norm_w, norm_h
                annotations.append(f"{class_id} {center_x} {center_y} {norm_w} {norm_h}")

    # Save annotation file only if annotations exist
    if annotations:
        annotation_filename = rgb_filename.replace("_lit.png", ".txt")
        annotation_path = os.path.join(dataset_dir, "labels/train", annotation_filename)
        with open(annotation_path, 'w') as f:
            f.write("\n".join(annotations))
        print(f"Annotation file {annotation_filename} created with {len(annotations)} entries.")

        # Copy RGB image to train folder
        rgb_output_path = os.path.join(dataset_dir, "images/train", rgb_filename)
        cv2.imwrite(rgb_output_path, cv2.imread(rgb_path))
        print(f"Copied RGB image {rgb_filename} to {rgb_output_path}")

# Final check if there are images in the train folder
image_files = [f for f in os.listdir(os.path.join(dataset_dir, "images/train")) if f.endswith(".png")]
if not image_files:
    print("No images found in the train directory. Ensure the data collection step was completed successfully.")
    sys.exit(1)

# Split the dataset into train, val, and test sets
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

# Move files to val and test folders
for val_file in val_files:
    if os.path.exists(os.path.join(dataset_dir, "images/train", val_file)):
        os.rename(
            os.path.join(dataset_dir, "images/train", val_file),
            os.path.join(dataset_dir, "images/val", val_file)
        )
    label_file = val_file.replace(".png", ".txt")
    if os.path.exists(os.path.join(dataset_dir, "labels/train", label_file)):
        os.rename(
            os.path.join(dataset_dir, "labels/train", label_file),
            os.path.join(dataset_dir, "labels/val", label_file)
        )

for test_file in test_files:
    if os.path.exists(os.path.join(dataset_dir, "images/train", test_file)):
        os.rename(
            os.path.join(dataset_dir, "images/train", test_file),
            os.path.join(dataset_dir, "images/test", test_file)
        )
    label_file = test_file.replace(".png", ".txt")
    if os.path.exists(os.path.join(dataset_dir, "labels/train", label_file)):
        os.rename(
            os.path.join(dataset_dir, "labels/train", label_file),
            os.path.join(dataset_dir, "labels/test", label_file)
        )

print("Data processing complete. Dataset ready for YOLO training.")