import os
import shutil
import random
from ultralytics import YOLO
import yaml
import torch

def copy_dataset(src_base_dir, dest_base_dir, weather_folders):
    """
    Copy RGB images and corresponding label files to the YOLO training directory structure.
    
    Args:
        src_base_dir (str): Source base directory containing weather folders with RGB images.
        dest_base_dir (str): Destination base directory for YOLO training data.
        weather_folders (list): List of weather folder names to process.
    """
    for weather in weather_folders:
        src_rgb_dir = os.path.join(src_base_dir, weather, "rgb")
        src_label_dir = os.path.join(src_base_dir, "dataset", "labels", "train", weather)
        
        dest_train_image_dir = os.path.join(dest_base_dir, "train", "images", weather)
        dest_train_label_dir = os.path.join(dest_base_dir, "train", "labels", weather)
        
        os.makedirs(dest_train_image_dir, exist_ok=True)
        os.makedirs(dest_train_label_dir, exist_ok=True)
        
        for file in os.listdir(src_rgb_dir):
            if file.endswith(".png"):
                image_src_path = os.path.join(src_rgb_dir, file)
                new_image_name = file.replace("_lit.png", ".png")
                label_file = new_image_name.replace(".png", ".txt")
                label_src_path = os.path.join(src_label_dir, label_file)
                
                if os.path.exists(label_src_path) and os.path.getsize(label_src_path) > 0:
                    shutil.copy(image_src_path, os.path.join(dest_train_image_dir, new_image_name))
                    shutil.copy(label_src_path, dest_train_label_dir)
                    print(f"Copied {file} as {new_image_name} and its label to {weather} training set.")
                else:
                    print(f"Skipping {file}: Corresponding label not found or empty.")

def split_train_val(dest_base_dir, split_ratio=0.8):
    """
    Split the dataset into training and validation sets.
    
    Args:
        dest_base_dir (str): Destination base directory for YOLO training data.
        split_ratio (float): Ratio of data to use for training (default is 0.8).
    """
    for subset in ['train', 'val']:
        images_dir = os.path.join(dest_base_dir, subset, "images")
        labels_dir = os.path.join(dest_base_dir, subset, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
    
    train_images_dir = os.path.join(dest_base_dir, "train", "images")
    train_labels_dir = os.path.join(dest_base_dir, "train", "labels")
    val_images_dir = os.path.join(dest_base_dir, "val", "images")
    val_labels_dir = os.path.join(dest_base_dir, "val", "labels")
    
    weather_folders = os.listdir(train_images_dir)
    
    for weather in weather_folders:
        weather_train_images = os.path.join(train_images_dir, weather)
        weather_train_labels = os.path.join(train_labels_dir, weather)
        
        weather_val_images = os.path.join(val_images_dir, weather)
        weather_val_labels = os.path.join(val_labels_dir, weather)
        
        os.makedirs(weather_val_images, exist_ok=True)
        os.makedirs(weather_val_labels, exist_ok=True)
        
        image_files = [f for f in os.listdir(weather_train_images) if f.endswith(".png")]
        random.shuffle(image_files)
        split_idx = int(len(image_files) * split_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        for file in val_files:
            label_file = file.replace(".png", ".txt")
            shutil.copy(os.path.join(weather_train_images, file), os.path.join(weather_val_images, file))
            shutil.copy(os.path.join(weather_train_labels, label_file), os.path.join(weather_val_labels, label_file))
            print(f"Copied {file} and its label to {weather} validation set.")
    
    print("Train/Validation split completed.")

def create_data_yaml(dest_base_dir, output_path):
    """
    Create the data.yaml file required by YOLOv8 for training.
    
    Args:
        dest_base_dir (str): Destination base directory for YOLO training data.
        output_path (str): Path to save the generated data.yaml file.
    """
    data_yaml = {
        'train': os.path.join(dest_base_dir, 'train', 'images').replace('\\', '/'),
        'val': os.path.join(dest_base_dir, 'val', 'images').replace('\\', '/'),
        'nc': 1,
        'names': ['DJIOctocopter']
    }
    with open(output_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print(f"data.yaml created at {output_path}")

def train_yolo_model(data_yaml_path, model_save_dir):
    """
    Train the YOLOv8 model using the prepared dataset.
    
    Args:
        data_yaml_path (str): Path to the data.yaml file.
        model_save_dir (str): Directory to save the trained model.
    """
    # Check if CUDA is available and use it if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = YOLO("yolov8n.pt", device=device)  # Using YOLOv8 Nano model
    model.train(data=data_yaml_path, epochs=50, imgsz=640, batch=16, name="yolov8_dji_model", save=True)
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
    
    # The results object should handle saving the video if the save argument is True
    print(f"Inference complete. Output saved at {output_video_path}")

def main():
    src_base_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput"
    dest_base_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\ObjectDetection\Yolo_train_model"
    weather_folders = ["clearsky", "sunnyclear", "sunnycloudy", "sunnycloudy2"]
    
    # Step 1: Copy dataset
    copy_dataset(src_base_dir, dest_base_dir, weather_folders)
    
    # Step 2: Split into train and validation sets
    split_train_val(dest_base_dir, split_ratio=0.8)
    
    # Step 3: Create data.yaml
    data_yaml_path = os.path.join(dest_base_dir, "data.yaml")
    create_data_yaml(dest_base_dir, data_yaml_path)
    
    # Step 4: Train YOLOv8 model
    model_save_dir = os.path.join(dest_base_dir, "runs", "train", "yolov8_dji_model")
    train_yolo_model(data_yaml_path, model_save_dir)
    
    # Step 5: Test the trained model
    model_path = os.path.join(model_save_dir, "weights", "best.pt")

    # model_path = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\runs\detect\yolov8_dji_model2\weights\best.pt"
    # test_video_path = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\dataset\images\test\MVI_3924 - Trim.MP4"
    # output_video_path = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\ObjectDetection\Yolo_train_model\output_video.mp4"
    # test_model_video(model_path, test_video_path, output_video_path)
    
    # step 6: test the model on the test dataset


    

if __name__ == "__main__":
    main() 