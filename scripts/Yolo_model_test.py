import os
import shutil
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch

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

def prepare_test_dataset(image_dir, annotation_dir, output_image_dir, output_label_dir, image_ranges):
    """
    Prepare the test dataset by converting XML annotations to YOLO format and saving images and labels.
    
    Args:
        image_dir (str): Directory containing the test images.
        annotation_dir (str): Directory containing the XML annotations.
        output_image_dir (str): Directory to save the test images.
        output_label_dir (str): Directory to save the YOLO formatted labels.
        image_ranges (list): List of tuples specifying the image ranges to include.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    total_images = 0

    for start, end in image_ranges:
        for i in range(start, end + 1):
            image_file = f"010{i:04d}.jpg"
            xml_file = f"010{i:04d}.xml"

            image_path = os.path.join(image_dir, image_file)
            xml_path = os.path.join(annotation_dir, xml_file)
            label_path = os.path.join(output_label_dir, f"010{i:04d}.txt")

            if os.path.exists(image_path) and os.path.exists(xml_path):
                # Copy the image to the output directory
                shutil.copy(image_path, os.path.join(output_image_dir, image_file))

                # Assuming all images have the same dimensions
                image_width, image_height = 3840, 2160
                yolo_annotations = convert_xml_to_yolo(xml_path, image_width, image_height)

                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_annotations))
                print(f"Processed {image_file} and saved annotations to {label_path}")
                total_images += 1
            else:
                print(f"Missing image or annotation for {i:07d}")

    print(f"Total images prepared for testing: {total_images}")

def test_model_images(model_path, test_images_dir, test_labels_dir, output_dir):
    """
    Test the trained YOLOv8 model on a test image dataset and evaluate performance metrics.
    
    Args:
        model_path (str): Path to the trained YOLOv8 model.
        test_images_dir (str): Path to the test images directory.
        test_labels_dir (str): Path to the test labels directory.
        output_dir (str): Directory to save the output results and metrics.
    """
    # Create a temporary data.yaml file for validation
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    data_yaml = {
        'train': '',  # Placeholder for train path
        'val': test_images_dir,
        'names': ['UAV']
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print(f"Temporary data.yaml created at {data_yaml_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = YOLO(model_path)
    results = model.val(
        data=data_yaml_path,
        save_json=True,
        save_dir=output_dir,
        device=device,
        plots=True
    )
    
    # Access the evaluation metrics directly
    metrics = results.box  # Assuming 'box' contains the metrics
    # print(f"Evaluation Metrics: {metrics}")
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(str(metrics))
    print(f"Metrics saved at {os.path.join(output_dir, 'metrics.txt')}")

    # Log metrics to TensorBoard
    writer = SummaryWriter(log_dir=output_dir)
    for key, value in metrics.items():
        writer.add_scalar(key, value, 0)
    writer.close()

def main():
    model_path = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\runs\detect\yolov8_SIDG-ATRID_trained_model\weights\best.pt"
    test_images_dir = r"D:\SiDG-ATRID-Dataset\Realworld-drone\DetFly\images\010"
    annotation_dir = r"D:\SiDG-ATRID-Dataset\Realworld-drone\DetFly\annotation\010"
    output_image_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\test_set\images"
    output_label_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\test_set\labels"
    output_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\ObjectDetection_yolo_dataset\Yolo_train_model\test_results"

    # Prepare the test dataset
    image_ranges = [(205, 398), (441, 550), (1265, 1349)]
    prepare_test_dataset(test_images_dir, annotation_dir, output_image_dir, output_label_dir, image_ranges)

    # Test the model
    test_model_images(model_path, output_image_dir, output_label_dir, output_dir)

if __name__ == "__main__":
    main() 