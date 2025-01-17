import os
import shutil
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch
import numpy as np
from ultralytics.utils.plotting import Annotator
from PIL import Image

def convert_xml_to_yolo(xml_file, image_width, image_height):
    """
    Convert XML annotation to YOLO format.
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

def prepare_test_dataset(image_dir, annotation_dir, output_image_dir, output_label_dir):
    """
    Prepare the test dataset by converting XML annotations to YOLO format and saving images and labels.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    total_images = 0
    positive_images = 0

    # Get all image and annotation filenames
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    annotation_files = set(f.lower() for f in os.listdir(annotation_dir) if f.lower().endswith('.xml'))

    # Match images with annotations based on a shared naming convention
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        annotation_file = f"{base_name}.xml".lower()

        # Copy image to output directory
        shutil.copy(os.path.join(image_dir, image_file), 
                   os.path.join(output_image_dir, image_file))
        
        # Create label file (empty for negative images)
        label_path = os.path.join(output_label_dir, f"{base_name}.txt")
        
        if annotation_file in annotation_files:
            # Positive image with drone - convert XML to YOLO format
            annotation_path = os.path.join(annotation_dir, f"{base_name}.xml")
            image_width, image_height = 640, 480
            yolo_annotations = convert_xml_to_yolo(annotation_path, image_width, image_height)
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_annotations))
            positive_images += 1
            print(f"Processed positive image: {image_file}")
        else:
            # Negative image without drone - create empty label file
            open(label_path, "w").close()
            print(f"Processed negative image: {image_file}")
        
        total_images += 1

    print(f"Total images prepared for testing: {total_images}")
    print(f"Positive images (with drones): {positive_images}")
    print(f"Negative images (without drones): {total_images - positive_images}")

def test_model_images(model_path, test_images_dir, test_labels_dir, output_dir):
    """
    Test the trained YOLOv8 model on a test image dataset and evaluate performance metrics.
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

    # Instead of results.metrics, use:
    metrics = results.box  # Contains the detection metrics
    maps = results.maps   # Dictionary of mAP values
    fitness = results.fitness # Fitness score

    # Extract key metrics
    metrics_dict = {
        'mAP50': float(results.box.map50),  # mAP at IoU 0.5
        'mAP50-95': float(results.box.map),  # mAP at IoU 0.5-0.95
        'Precision': float(results.box.mp),  # Mean Precision
        'Recall': float(results.box.mr),     # Mean Recall
        'F1-Score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-16),  # F1 Score
    }

    # If available, add per-class metrics
    if hasattr(results.box, 'p') and len(results.box.p) > 0:
        for i, name in enumerate(results.names):
            try:
                p, r, ap50, ap = results.box.class_result(i)
                metrics_dict.update({
                    f'Class_{name}_Precision': float(p),
                    f'Class_{name}_Recall': float(r),
                    f'Class_{name}_AP50': float(ap50),
                    f'Class_{name}_AP50-95': float(ap),
                })
            except IndexError:
                continue

    # Save metrics to CSV
    import pandas as pd
    metrics_df = pd.DataFrame([metrics_dict])
    csv_path = os.path.join(output_dir, 'performance_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Performance metrics saved to: {csv_path}")

    # Also save as a formatted text file for easy reading
    txt_path = os.path.join(output_dir, 'performance_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("Performance Metrics Summary\n")
        f.write("==========================\n\n")
        for metric, value in metrics_dict.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"Performance summary saved to: {txt_path}")

    # # Writing metrics to file
    # metrics_file = os.path.join(output_dir, "metrics.txt")
    # with open(metrics_file, "w") as f:
    #     f.write(f"Mean Results: {results.mean_results()}\n")  # Gets mean values of metrics
    #     f.write(f"mAP Values: {results.maps}\n")
    #     f.write(f"Fitness Score: {results.fitness}\n")
        
    #     # Safely handle class-specific results
    #     if hasattr(results.box, 'p') and len(results.box.p) > 0:
    #         for i, name in enumerate(results.names):
    #             try:
    #                 class_result = results.class_result(i)
    #                 f.write(f"Class {name} Results: {class_result}\n")
    #             except IndexError:
    #                 f.write(f"Class {name} Results: No detections\n")
    # Log metrics to TensorBoard
    writer = SummaryWriter(log_dir=output_dir)

    # Log method-based metrics
    writer.add_scalar('Mean Precision', metrics.mp, 0)
    writer.add_scalar('Mean Recall', metrics.mr, 0)
    writer.add_scalar('mAP@0.5', metrics.map50, 0)
    writer.add_scalar('mAP@0.75', metrics.map75, 0)
    writer.add_scalar('mAP@0.5:0.95', metrics.map, 0)
    writer.add_scalar('Model Fitness', metrics.fitness(), 0)

    writer.close()
    print(f"Metrics logged to TensorBoard at {output_dir}")

def visualize_predictions(model, test_images_dir, output_image_dir):
    """
    Visualize predictions on test images and save annotated images.
    """
    os.makedirs(output_image_dir, exist_ok=True)

    for image_file in os.listdir(test_images_dir):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(test_images_dir, image_file)
        try:
            # Load the image
            image = Image.open(image_path)
            image_np = np.array(image)

            # Run the YOLO model on the image
            results = model(image_path)
            annotator = Annotator(image_np)

            # Add bounding boxes and labels
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                scores = result.boxes.conf.cpu().numpy()  # Get confidence scores
                for box, score in zip(boxes, scores):
                    annotator.box_label(box, f'UAV {score:.2f}')  # Label format: UAV <confidence>

            # Save the annotated image
            output_image_path = os.path.join(output_image_dir, f"annotated_{image_file}")
            annotator.save(output_image_path)
            print(f"Saved annotated image: {output_image_path}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def main():
    model_path = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\models\warsaw-dronedetect-model\weights\best.pt"
    test_images_dir = r"D:\SiDG-ATRID-Dataset\Warsaw-DroneDetectionDataset\DroneTestDataset\Drone_TestSet"
    annotation_dir = r"D:\SiDG-ATRID-Dataset\Warsaw-DroneDetectionDataset\DroneTestDataset\Drone_TestSet_XMLs"
    output_image_dir = r"D:\SiDG-ATRID-Dataset\runs\detect\Warsaw-Yolo_DroneDetectiontest_model\images"
    output_label_dir = r"D:\SiDG-ATRID-Dataset\runs\detect\Warsaw-Yolo_DroneDetectiontest_model\labels"
    output_dir = r"D:\SiDG-ATRID-Dataset\runs\detect\Warsaw-Yolo_DroneDetectiontest_model"

    # Prepare the test dataset
    prepare_test_dataset(test_images_dir, annotation_dir, output_image_dir, output_label_dir)

    # Test the model
    test_model_images(model_path, output_image_dir, output_label_dir, output_dir)

    # Visualize predictions
    model = YOLO(model_path)
    visualize_predictions(model, output_image_dir, output_image_dir)

if __name__ == "__main__":
    main()
