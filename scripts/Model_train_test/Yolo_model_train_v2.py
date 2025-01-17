"""
Title: YOLO Model Training Script v2
Author: Joshua Chang
Description: This script is designed to facilitate the training of a YOLOv8 model using a specified dataset. 
It includes functionalities for copying datasets, creating necessary configuration files, and executing the 
training process. The script is structured to handle both real-world and simulated data, allowing for flexible 
dataset management and model training. The main function orchestrates the entire workflow, ensuring that each 
step is executed in sequence, from data preparation to model training.
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import yaml
import torch
import argparse

def convert_xml_to_yolo(xml_file, image_width, image_height):
    """
    Convert XML annotation to YOLO format.
    
    Args:
        xml_file (str): Path to the XML file.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    
    Returns:
        list: A list of YOLO formatted annotations.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_annotations = []

    for obj in root.findall('object'):
        class_id = 0  # Assuming 'UAV' is the only class and is indexed as 0
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return yolo_annotations

def copy_dataset_sim(src_base_dir, dest_base_dir, weather_folders, num_images=None, selected_folders=None):
    """
    Copy RGB images and corresponding label files to the YOLO training directory structure for simulated data.
    
    Args:
        src_base_dir (str): Source base directory containing weather folders with RGB images.
        dest_base_dir (str): Destination base directory for YOLO training data.
        weather_folders (list): List of weather folder names to process.
        num_images (int, optional): Total number of images to copy from simulated data. If None, copy all.
        selected_folders (list, optional): Specific weather folders to include. If None, include all.
    """
    copied_images = 0
    for weather in weather_folders:
        if selected_folders and weather not in selected_folders:
            continue  # Skip if not in selected folders
        src_rgb_dir = os.path.join(src_base_dir, weather, "rgb")
        src_label_dir = os.path.join(src_base_dir, "dataset", "labels", "train", weather)
        
        dest_train_image_dir = os.path.join(dest_base_dir, "train", "images")
        dest_train_label_dir = os.path.join(dest_base_dir, "train", "labels")
        
        os.makedirs(dest_train_image_dir, exist_ok=True)
        os.makedirs(dest_train_label_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(src_rgb_dir) if f.endswith(".png")]
        random.shuffle(image_files)
        
        for file in image_files:
            if file.endswith(".png"):
                image_src_path = os.path.join(src_rgb_dir, file)
                new_image_name = file.replace("_lit.png", ".png")
                label_file = new_image_name.replace(".png", ".txt")
                label_src_path = os.path.join(src_label_dir, label_file)
                
                if os.path.exists(label_src_path) and os.path.getsize(label_src_path) > 0:
                    shutil.copy(image_src_path, os.path.join(dest_train_image_dir, new_image_name))
                    shutil.copy(label_src_path, dest_train_label_dir)
                    print(f"Copied {file} as {new_image_name} and its label to simulated training set.")
                    copied_images += 1
                    if num_images and copied_images >= num_images:
                        print(f"Reached the specified number of simulated images: {num_images}")
                        return copied_images
                else:
                    print(f"Skipping {file}: Corresponding label not found or empty.")
    return copied_images

def copy_dataset_real(src_image_dir, src_label_dir, dest_base_dir, num_images=None, image_width=3840, image_height=2160):
    """
    Split the real-world dataset into training and validation sets and copy the image and label pairs.
    Converts XML annotations to YOLO format.
    
    Args:
        src_image_dir (str): Source directory containing images.
        src_label_dir (str): Source directory containing XML annotations.
        dest_base_dir (str): Destination base directory for YOLO training data.
        num_images (int, optional): Total number of images to copy from real-world data. If None, copy all.
        image_width (int): Width of the images (assumed to be consistent).
        image_height (int): Height of the images (assumed to be consistent).
    """
    dest_train_image_dir = os.path.join(dest_base_dir, "train", "images")
    dest_train_label_dir = os.path.join(dest_base_dir, "train", "labels")
    
    os.makedirs(dest_train_image_dir, exist_ok=True)
    os.makedirs(dest_train_label_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(src_image_dir) if f.endswith(".jpg")]
    random.shuffle(image_files)
    
    if num_images:
        image_files = image_files[:num_images]
    
    copied_images = 0
    for file in image_files:
        if file.endswith(".jpg"):
            image_src_path = os.path.join(src_image_dir, file)
            base_name = os.path.splitext(file)[0]  # Extract base name without extension
            
            # Corresponding XML file
            xml_file = f"{base_name}.xml"
            xml_src_path = os.path.join(src_label_dir, xml_file)
            
            if os.path.exists(xml_src_path):
                # Convert XML to YOLO format
                yolo_annotations = convert_xml_to_yolo(xml_src_path, image_width, image_height)
                
                if yolo_annotations:
                    # Save YOLO annotations to TXT file
                    yolo_label_path = os.path.join(dest_train_label_dir, f"{base_name}.txt")
                    with open(yolo_label_path, "w") as f:
                        f.write("\n".join(yolo_annotations))
                    
                    # Copy the image to the destination
                    shutil.copy(image_src_path, os.path.join(dest_train_image_dir, file))
                    print(f"Copied Image: {file} and converted its XML annotation to YOLO format.")
                    copied_images += 1
                else:
                    print(f"No valid annotations found in {xml_file}. Skipping.")
            else:
                print(f"Skipping {file}: Corresponding XML annotation not found.")
    return copied_images

def copy_dataset_both(sim_src_base_dir, real_src_image_dir, real_src_label_dir, dest_base_dir, total_images, sim_ratio=0.5, image_width=3840, image_height=2160):
    """
    Combine simulated and real-world datasets based on specified ratio and total images.
    
    Args:
        sim_src_base_dir (str): Source base directory for simulated data.
        real_src_image_dir (str): Source directory for real-world images.
        real_src_label_dir (str): Source directory for real-world labels.
        dest_base_dir (str): Destination base directory for YOLO training data.
        total_images (int): Total number of images to include in the combined dataset.
        sim_ratio (float): Ratio of simulated data in the combined dataset (default is 0.5).
        image_width (int): Width of the images (assumed to be consistent).
        image_height (int): Height of the images (assumed to be consistent).
    """
    sim_images_num = int(total_images * sim_ratio)
    real_images_num = total_images - sim_images_num
    
    print(f"Preparing combined dataset with {sim_images_num} simulated and {real_images_num} real-world images.")
    
    # Copy simulated data
    copied_sim = copy_dataset_sim(
        src_base_dir=sim_src_base_dir,
        dest_base_dir=dest_base_dir,
        weather_folders=["clearsky", "sunnyclear", "sunnycloudy", "sunnycloudy2"],
        num_images=sim_images_num
    )
    
    # Copy real-world data
    copied_real = copy_dataset_real(
        src_image_dir=real_src_image_dir,
        src_label_dir=real_src_label_dir,
        dest_base_dir=dest_base_dir,
        num_images=real_images_num,
        image_width=image_width,
        image_height=image_height
    )
    
    print(f"Copied {copied_sim} simulated and {copied_real} real-world images to the combined dataset.")

def split_train_val(dest_base_dir, split_ratio=0.8):
    """
    Split the dataset into training and validation sets.
    
    Args:
        dest_base_dir (str): Destination base directory for YOLO training data.
        split_ratio (float): Ratio of data to use for training (default is 0.8).
    """
    # Create validation directories
    val_image_dir = os.path.join(dest_base_dir, "val", "images")
    val_label_dir = os.path.join(dest_base_dir, "val", "labels")
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Get all image files in train/images
    train_image_dir = os.path.join(dest_base_dir, "train", "images")
    train_label_dir = os.path.join(dest_base_dir, "train", "labels")
    
    image_files = [f for f in os.listdir(train_image_dir) if f.endswith((".png", ".jpg"))]
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Move validation files
    for file in val_files:
        # Move image
        src_image = os.path.join(train_image_dir, file)
        dest_image = os.path.join(val_image_dir, file)
        shutil.move(src_image, dest_image)
        
        # Move label
        label_file = os.path.splitext(file)[0] + ".txt"
        src_label = os.path.join(train_label_dir, label_file)
        dest_label = os.path.join(val_label_dir, label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, dest_label)
            print(f"Moved {file} and its label to validation set.")
        else:
            print(f"Label file for {file} not found. Skipping label move.")
    
    print(f"Dataset split into {len(train_files)} training and {len(val_files)} validation samples.")

def create_data_yaml(dest_base_dir, output_path, names, nc=1):
    """
    Create the data.yaml file required by YOLOv8 for training.
    
    Args:
        dest_base_dir (str): Destination base directory for YOLO training data.
        output_path (str): Path to save the generated data.yaml file.
        names (list): List of class names.
        nc (int): Number of classes.
    """
    data_yaml = {
        'train': os.path.join(dest_base_dir, 'train', 'images').replace('\\', '/'),
        'val': os.path.join(dest_base_dir, 'val', 'images').replace('\\', '/'),
        'nc': nc,
        'names': names
    }
    with open(output_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print(f"data.yaml created at {output_path}")

def train_yolo_model(data_yaml_path, model_save_dir, device, model_name):
    """
    Train the YOLOv8 model using the prepared dataset.
    
    Args:
        data_yaml_path (str): Path to the data.yaml file.
        model_save_dir (str): Directory to save the trained model.
        device (str): Device to use for training ('cuda' or 'cpu').
        model_name (str): Name of the training run/model.
    """
    model = YOLO("yolov8s.pt")  # Using YOLOv8s 
    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name=model_name,
        project=model_save_dir,  # Set the project directory to save the model
        save=True,
        cache=False  # Ensure cache is disabled
    )
    print(f"Training complete. Model saved in {model_save_dir}")

def test_model_video(model_path, test_video_path, output_video_path):
    """
    Test the trained YOLOv8 model on a test video.
    
    Args:
        model_path (str): Path to the trained YOLOv8 model.
        test_video_path (str): Path to the test video file.
        output_video_path (str): Path to save the output video with detections.
    """
    model = YOLO(model_path)
    results = model.predict(
        source=test_video_path,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_boxes=True,
        show_conf=True
    )
    
    # The results object handles saving the video if the save argument is True
    print(f"Inference complete. Output saved at {output_video_path}")

def main():
    # Default parameters (can be overridden by command-line arguments)
    DEFAULT_MODE = 'real'  # 'sim', 'real', or 'both'
    DEFAULT_TOTAL_IMAGES = 4000  # Only used in 'both' mode
    DEFAULT_SIM_RATIO = 0.5  # Ratio of sim data in 'both' mode
    DEFAULT_MODEL_NAME = "warsaw-dronedetect-model"
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script for Simulated, Real-World, and Combined Data")
    parser.add_argument('--mode', type=str, choices=['sim', 'real', 'both'], help="Mode of operation: 'sim', 'real', or 'both'")
    parser.add_argument('--total_images', type=int, help="Total number of images to use in 'both' mode")
    parser.add_argument('--sim_ratio', type=float, help="Ratio of simulated data in 'both' mode (default is 0.5)")
    parser.add_argument('--model_name', type=str, help="Name of the YOLOv8 training run/model")
    
    args = parser.parse_args()
    
    # Set parameters based on arguments or defaults
    mode = args.mode if args.mode else DEFAULT_MODE
    total_images = args.total_images if args.total_images else DEFAULT_TOTAL_IMAGES
    sim_ratio = args.sim_ratio if args.sim_ratio else DEFAULT_SIM_RATIO
    model_name = args.model_name if args.model_name else DEFAULT_MODEL_NAME
    
    print(f"=== YOLOv8 Training Script ===")
    print(f"Mode: {mode}")
    if mode == 'both':
        print(f"Total Images: {total_images}")
        print(f"Simulated Data Ratio: {sim_ratio}")
    print(f"Model Name: {model_name}\n")
    
    if mode == 'sim':
        print("=== Running in Simulated Data Mode ===")
        # Parameters for simulated data
        sim_src_base_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput"
        dest_base_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\ObjectDetection\Yolo_train_model_simulated"
        weather_folders = ["clearsky", "sunnyclear", "sunnycloudy", "sunnycloudy2"]
        
        # Step 1: Copy dataset for simulated data
        print("Step 1: Copying simulated dataset...")
        copy_dataset_sim(
            src_base_dir=sim_src_base_dir,
            dest_base_dir=dest_base_dir,
            weather_folders=weather_folders,
            num_images=None  # Copy all simulated images
        )
        
        # Step 2: Split into train and validation sets for simulated data
        print("Step 2: Splitting simulated dataset into train and val...")
        split_train_val(dest_base_dir, split_ratio=0.8)
        
        # Step 3: Create data.yaml for simulated data
        print("Step 3: Creating data.yaml for simulated data...")
        data_yaml_path = os.path.join(dest_base_dir, "data.yaml")
        create_data_yaml(dest_base_dir, data_yaml_path, names=['DJIOctocopter'], nc=1)
        
        # Step 4: Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}\n")
        
        # Step 5: Train YOLOv8 model for simulated data
        print("Step 4: Training YOLOv8 model on simulated data...")
        model_save_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\models"  # Updated save directory
        train_yolo_model(data_yaml_path, model_save_dir, device, model_name)
        
    elif mode == 'real':
        print("=== Running in Real-World Data Mode ===")
        # Parameters for real-world data
        real_src_image_dir = r"D:\SiDG-ATRID-Dataset\Warsaw-DroneDetectionDataset\DroneTrainDataset\Drone_TrainSet"
        real_src_label_dir = r"D:\SiDG-ATRID-Dataset\Warsaw-DroneDetectionDataset\DroneTrainDataset\Drone_TrainSet_XMLs"
        dest_base_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\models\Warsaw-DroneDetection_Dataset\processed_yolo_images_labels"
        
        # Step 1: Copy and split dataset for real-world data
        print("Step 1: Copying and preparing real-world dataset...")
        copy_dataset_real(
            src_image_dir=real_src_image_dir,
            src_label_dir=real_src_label_dir,
            dest_base_dir=dest_base_dir,
            num_images=None,  # Copy all real-world images
            image_width=640,  # Update if your images have different dimensions
            image_height=480
        )
        
        # Step 2: Split into train and validation sets for real-world data
        print("Step 2: Splitting real-world dataset into train and val...")
        split_train_val(dest_base_dir, split_ratio=0.8)
        
        # Step 3: Create data.yaml for real-world data
        print("Step 3: Creating data.yaml for real-world data...")
        data_yaml_path = os.path.join(dest_base_dir, "data.yaml")
        create_data_yaml(dest_base_dir, data_yaml_path, names=['UAV'], nc=1)
        
        # Step 4: Determine device and set CUDA device if available
        if torch.cuda.is_available():
            device = 'cuda'
            try:
                torch.cuda.set_device(0)
                print("CUDA is available. Set to GPU 0.")
            except Exception as e:
                print(f"Error setting CUDA device: {e}")
                device = 'cpu'
        else:
            device = 'cpu'
            print("CUDA is not available. Using CPU.")
        print(f"Using device: {device}\n")
        
        # Step 5: Train YOLOv8 model for real-world data
        print("Step 4: Training YOLOv8 model on real-world data...")
        model_save_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\models\Warsaw-DroneDetection_Dataset"  # Updated save directory
        train_yolo_model(data_yaml_path, model_save_dir, device, model_name)
        
    elif mode == 'both':
        print("=== Running in Combined (Simulated + Real-World) Data Mode ===")
        # Parameters for combined data
        sim_src_base_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput"
        real_src_image_dir = r"D:\SiDG-ATRID-Dataset\Warsaw-DroneDetectionDataset\DroneTrainDataset\Drone_TrainSet"
        real_src_label_dir = r"D:\SiDG-ATRID-Dataset\Warsaw-DroneDetectionDataset\DroneTrainDataset\Drone_TrainSet_XMLs"
        dest_base_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\ObjectDetection_yolo_dataset\Yolo_train_model_combined"
        
        # Step 1: Copy and combine datasets
        print("Step 1: Copying and combining simulated and real-world datasets...")
        copy_dataset_both(
            sim_src_base_dir=sim_src_base_dir,
            real_src_image_dir=real_src_image_dir,
            real_src_label_dir=real_src_label_dir,
            dest_base_dir=dest_base_dir,
            total_images=total_images,
            sim_ratio=sim_ratio,
            image_width=3840,  # Update if your images have different dimensions
            image_height=2160
        )
        
        # Step 2: Split into train and validation sets for combined data
        print("Step 2: Splitting combined dataset into train and val...")
        split_train_val(dest_base_dir, split_ratio=0.8)
        
        # Step 3: Create data.yaml for combined data
        print("Step 3: Creating data.yaml for combined data...")
        data_yaml_path = os.path.join(dest_base_dir, "data.yaml")
        create_data_yaml(dest_base_dir, data_yaml_path, names=['DJIOctocopter', 'UAV'], nc=2)
        
        # Step 4: Determine device and set CUDA device if available
        if torch.cuda.is_available():
            device = 'cuda'
            try:
                torch.cuda.set_device(0)
                print("CUDA is available. Set to GPU 0.")
            except Exception as e:
                print(f"Error setting CUDA device: {e}")
                device = 'cpu'
        else:
            device = 'cpu'
            print("CUDA is not available. Using CPU.")
        print(f"Using device: {device}\n")
        
        # Step 5: Train YOLOv8 model for combined data
        print("Step 4: Training YOLOv8 model on combined data...")
        model_save_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\models"  # Updated save directory
        train_yolo_model(data_yaml_path, model_save_dir, device, model_name)
        
    else:
        print("Invalid mode selected. Choose either 'sim', 'real', or 'both'.")
        return
    
    # Optional: Test the trained model
    # Uncomment and configure the following lines as needed.
    """
    # Example usage for testing
    if mode in ['sim', 'both']:
        model_path = os.path.join(model_save_dir, "weights", "best.pt")
        test_video_path = r"path_to_test_video.mp4"
        output_video_path = r"path_to_output_video.mp4"
        print("Testing the trained model on a video...")
        test_model_video(model_path, test_video_path, output_video_path)
    elif mode == 'real':
        model_path = os.path.join(model_save_dir, "weights", "best.pt")
        test_video_path = r"path_to_test_video.mp4"
        output_video_path = r"path_to_output_video.mp4"
        print("Testing the trained model on a video...")
        test_model_video(model_path, test_video_path, output_video_path)
    """
    
    print("=== YOLOv8 Training Workflow Completed ===")

if __name__ == "__main__":
    main()
