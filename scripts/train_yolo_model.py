import os
import random
import shutil
from ultralytics import YOLO


def prepare_dataset_structure():
    """
    Prepare a YOLO-compatible dataset by combining RGB images and their corresponding
    bounding box `.txt` files into a single directory structure.
    """
    # Source directories
    rgb_dirs = [
        r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\clearsky\rgb",
        r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\nightcloudy\rgb",
        r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\sunnyclear\rgb",
        r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\sunnycloudy\rgb",
    ]
    label_dirs = [
        r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\dataset\labels\train\clearsky",
        r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\dataset\labels\train\nightcloudy",
        r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\dataset\labels\train\sunnyclear",
        r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\dataset\labels\train\sunnycloudy",
    ]

    # Output directories
    dataset_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\prepared_dataset"
    train_image_dir = os.path.join(dataset_dir, "train", "images")
    train_label_dir = os.path.join(dataset_dir, "train", "labels")

    # Create directories
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)

    # Process RGB and label files
    total_files = 0
    for rgb_dir, label_dir in zip(rgb_dirs, label_dirs):
        for file in os.listdir(rgb_dir):
            if file.endswith(".png"):
                image_path = os.path.join(rgb_dir, file)
                label_file_name = file.replace("_lit.png", ".txt")
                label_path = os.path.join(label_dir, label_file_name)

                # Check if the corresponding label file exists and is non-empty
                if not os.path.exists(label_path):
                    print(f"Skipping {file}: No corresponding label file found.")
                    continue
                if os.path.getsize(label_path) == 0:
                    print(f"Skipping {file}: Label file is empty.")
                    continue

                # Move image and label to YOLO dataset structure
                dest_image_path = os.path.join(train_image_dir, file)
                dest_label_path = os.path.join(train_label_dir, label_file_name)

                try:
                    shutil.copy(image_path, dest_image_path)
                    shutil.copy(label_path, dest_label_path)
                    total_files += 1
                except Exception as e:
                    print(f"Error moving {file}: {e}")

    print(f"Dataset prepared with {total_files} images and labels in {train_image_dir} and {train_label_dir}.")


def split_train_val():
    """
    Split 80% of the dataset into train and 20% into validation.
    Moves files from prepared_dataset/train to prepared_dataset/val.
    """
    # Directories
    base_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\prepared_dataset"
    train_image_dir = os.path.join(base_dir, "train", "images")
    train_label_dir = os.path.join(base_dir, "train", "labels")
    val_image_dir = os.path.join(base_dir, "val", "images")
    val_label_dir = os.path.join(base_dir, "val", "labels")

    # Create val directories if they don't exist
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Collect all image-label pairs
    images = [f for f in os.listdir(train_image_dir) if f.endswith(".png")]
    labels = [f.replace("_lit.png", ".txt") for f in images]

    # Ensure all labels exist
    valid_pairs = []
    for img, lbl in zip(images, labels):
        if os.path.exists(os.path.join(train_label_dir, lbl)):
            valid_pairs.append((img, lbl))

    # Shuffle and split
    random.shuffle(valid_pairs)
    split_idx = int(0.8 * len(valid_pairs))  # 80% for training, 20% for validation
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    # Move val files to val directory
    for img, lbl in val_pairs:
        shutil.move(os.path.join(train_image_dir, img), os.path.join(val_image_dir, img))
        shutil.move(os.path.join(train_label_dir, lbl), os.path.join(val_label_dir, lbl))
        print(f"Moved {img} and {lbl} to validation set.")

    print(f"Split complete: {len(train_pairs)} training and {len(val_pairs)} validation pairs.")
    print(f"Train images: {len(os.listdir(train_image_dir))}, Train labels: {len(os.listdir(train_label_dir))}")
    print(f"Validation images: {len(os.listdir(val_image_dir))}, Validation labels: {len(os.listdir(val_label_dir))}")


def create_data_yaml():
    """
    Create the YOLO data.yaml file for training.
    """
    dataset_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\prepared_dataset"
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")

    # Use forward slashes for paths
    train_images_path = os.path.join(dataset_dir, 'train', 'images').replace("\\", "/")
    val_images_path = os.path.join(dataset_dir, 'val', 'images').replace("\\", "/")

    # Write data.yaml
    data_yaml = f"""
train: {train_images_path}
val: {val_images_path}
nc: 1
names: ['DJIOctocopter']
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml)
    print(f"data.yaml created at {data_yaml_path}")


def train_yolo_model():
    """
    Train the YOLOv8 model using the prepared dataset.
    """
    dataset_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\prepared_dataset"
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    
    # Train YOLOv8
    model = YOLO("yolov8n.pt")  # YOLOv8 Nano
    print("Starting YOLOv8 training...")
    model.train(data=data_yaml_path, epochs=50, imgsz=640, batch=16, name="yolov8_dji_model")
    print("Training complete.")


def verify_labels():
    """
    Verify that label files are correctly named and contain valid annotations.
    """
    train_label_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\prepared_dataset\train\labels"
    label_files = [f for f in os.listdir(train_label_dir) if f.endswith(".txt")]
    print(f"Found {len(label_files)} label files in {train_label_dir}. Checking for non-empty files.")

    for label_file in label_files[:5]:  # Sample check first 5 files
        label_path = os.path.join(train_label_dir, label_file)
        with open(label_path, "r") as file:
            content = file.read().strip()
        print(f"File: {label_file}, Content: {content}")


if __name__ == "__main__":
    # Step 1: Prepare the dataset structure
    prepare_dataset_structure()

    # Step 2: Split into train/val sets
    split_train_val()

    # Step 3: Verify labels
    verify_labels()

    # Step 4: Create the data.yaml file
    create_data_yaml()

    # Step 5: Train the YOLOv8 model
    train_yolo_model()
