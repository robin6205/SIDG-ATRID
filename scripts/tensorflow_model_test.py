"""
File: tensorflow_model_test.py
Description: This script is designed for testing TensorFlow models using a given test dataset.
Author: [Your Name]
Date: [Current Date]
"""
import tensorflow as tf
import os
import glob
import cv2
import numpy as np
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from torch.utils.tensorboard import SummaryWriter

def load_model(checkpoint_dir, config_path):
    """
    Load the TensorFlow object detection model from checkpoint.
    """
    configs = config_util.get_configs_from_pipeline_file(config_path)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(checkpoint_dir)).expect_partial()
    print("Model loaded successfully from checkpoint.")
    return detection_model

def run_inference(model, test_images_dir, output_dir, label_map_path, writer, step=0):
    """
    Run inference on test images, log results to TensorBoard, and save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    label_map = label_map_util.create_category_index_from_labelmap(label_map_path)

    test_images = glob.glob(os.path.join(test_images_dir, "*.jpg"))
    for i, image_path in enumerate(test_images):
        # Load and preprocess image
        image_np = cv2.imread(image_path)
        input_tensor = tf.convert_to_tensor(image_np[np.newaxis, ...], dtype=tf.float32)

        # Perform inference
        detections = model(input_tensor)

        # Visualize detections
        viz_image = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            viz_image,
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(int),
            detections['detection_scores'][0].numpy(),
            label_map,
            use_normalized_coordinates=True,
            line_thickness=3
        )

        # Save visualized image
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_image_path, viz_image)
        print(f"Saved detection output to {output_image_path}")

        # Log image to TensorBoard
        writer.add_image(
            f'Detection Results/{os.path.basename(image_path)}',
            viz_image,
            step + i,
            dataformats='HWC'
        )

# def evaluate_model(model, test_images_dir, annotation_dir, writer, step=0):
#     """
#     Evaluate the model and log metrics to TensorBoard.
#     """
#     # Placeholder for evaluation logic (e.g., mAP, precision, recall)
#     # Simulated metrics for demonstration
#     precision = 0.85
#     recall = 0.75
#     mAP = 0.80

#     # Log metrics to TensorBoard
#     writer.add_scalar('Metrics/Precision', precision, step)
#     writer.add_scalar('Metrics/Recall', recall, step)
#     writer.add_scalar('Metrics/mAP', mAP, step)
#     print(f"Logged evaluation metrics to TensorBoard.")

def evaluate_model(model, test_images_dir, annotation_dir, writer, output_dir, step=0):
    """
    Evaluate the model and save detailed performance metrics.
    """
    # Calculate actual metrics instead of using placeholders
    detections_list = []
    ground_truth_list = []
    
    test_images = glob.glob(os.path.join(test_images_dir, "*.jpg"))
    for image_path in test_images:
        # Get image name without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotation_path = os.path.join(annotation_dir, f"{base_name}.xml")
        
        # Load and preprocess image
        image_np = cv2.imread(image_path)
        input_tensor = tf.convert_to_tensor(image_np[np.newaxis, ...], dtype=tf.float32)
        
        # Perform detection
        detections = model(input_tensor)
        
        # Convert detections to a more manageable format
        detection_boxes = detections['detection_boxes'][0].numpy()
        detection_scores = detections['detection_scores'][0].numpy()
        detection_classes = detections['detection_classes'][0].numpy().astype(int)
        
        # Store detections and ground truth for metric calculation
        detections_list.append({
            'boxes': detection_boxes,
            'scores': detection_scores,
            'classes': detection_classes
        })
        
        # Parse ground truth from XML
        if os.path.exists(annotation_path):
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            gt_boxes = []
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                # Normalize coordinates
                width = float(root.find('size').find('width').text)
                height = float(root.find('size').find('height').text)
                gt_boxes.append([ymin/height, xmin/width, ymax/height, xmax/width])
            ground_truth_list.append({
                'boxes': np.array(gt_boxes),
                'classes': np.ones(len(gt_boxes), dtype=int)  # Assuming single class (UAV)
            })
        else:
            ground_truth_list.append({
                'boxes': np.array([]),
                'classes': np.array([])
            })

    # Calculate metrics
    metrics_dict = calculate_detection_metrics(detections_list, ground_truth_list)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics_dict])
    csv_path = os.path.join(output_dir, 'performance_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Performance metrics saved to: {csv_path}")

    # Save formatted summary
    txt_path = os.path.join(output_dir, 'performance_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("TensorFlow Object Detection Performance Metrics\n")
        f.write("==========================================\n\n")
        for metric, value in metrics_dict.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"Performance summary saved to: {txt_path}")

    # Log metrics to TensorBoard
    for metric, value in metrics_dict.items():
        writer.add_scalar(f'Metrics/{metric}', value, step)

def calculate_detection_metrics(detections_list, ground_truth_list, iou_threshold=0.5, score_threshold=0.5):
    """
    Calculate detection metrics including mAP, precision, recall, and F1 score.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for detections, ground_truth in zip(detections_list, ground_truth_list):
        # Filter detections by score threshold
        mask = detections['scores'] >= score_threshold
        det_boxes = detections['boxes'][mask]
        
        gt_boxes = ground_truth['boxes']
        
        if len(gt_boxes) == 0:
            # If no ground truth, all detections are false positives
            false_positives += len(det_boxes)
            continue
            
        if len(det_boxes) == 0:
            # If no detections, all ground truths are false negatives
            false_negatives += len(gt_boxes)
            continue
        
        # Calculate IoU matrix
        ious = calculate_iou_matrix(det_boxes, gt_boxes)
        
        # Match detections to ground truth boxes
        matched_gt = set()
        for i in range(len(det_boxes)):
            max_iou = np.max(ious[i])
            if max_iou >= iou_threshold:
                gt_idx = np.argmax(ious[i])
                if gt_idx not in matched_gt:
                    true_positives += 1
                    matched_gt.add(gt_idx)
                else:
                    false_positives += 1
            else:
                false_positives += 1
        
        # Count unmatched ground truth boxes as false negatives
        false_negatives += len(gt_boxes) - len(matched_gt)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'True_Positives': true_positives,
        'False_Positives': false_positives,
        'False_Negatives': false_negatives
    }

def calculate_iou_matrix(boxes1, boxes2):
    """
    Calculate IoU matrix between two sets of boxes.
    Boxes are in format [ymin, xmin, ymax, xmax].
    """
    # Calculate intersection areas
    y_min = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    x_min = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    y_max = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    x_max = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    
    intersection_heights = np.maximum(0., y_max - y_min)
    intersection_widths = np.maximum(0., x_max - x_min)
    intersection_areas = intersection_heights * intersection_widths
    
    # Calculate areas
    boxes1_areas = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_areas = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate union areas
    union_areas = (boxes1_areas[:, None] + boxes2_areas[None, :] - intersection_areas)
    
    # Calculate IoU
    iou_matrix = intersection_areas / (union_areas + 1e-6)
    
    return iou_matrix

def main():
    checkpoint_dir = r"D:\SiDG-ATRID-Dataset\Warsaw-DroneDetectionDataset\PaperBasedANNModels\droneInfGraph401092\model.ckpt"
    config_path = r"D:\SiDG-ATRID-Dataset\Warsaw-DroneDetectionDataset\PaperBasedANNModels\droneInfGraph401092\pipeline.config"
    test_images_dir = r"D:\SiDG-ATRID-Dataset\Warsaw-DroneDetectionDataset\DroneTestDataset\Drone_TestSet"
    annotation_dir = r"D:\SiDG-ATRID-Dataset\Warsaw-DroneDetectionDataset\DroneTestDataset\Drone_TestSet_XMLs"
    output_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\runs\Warsaw-DroneDetectionDataset_test\TestResults"
    label_map_path = r"D:\SiDG-ATRID-Dataset\Warsaw-DroneDetectionDataset\PaperBasedANNModels\droneInfGraph401092\object-detection.pbtxt"

    # Initialize TensorBoard writer
    log_dir = os.path.join(output_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)

    # Load model
    model = load_model(checkpoint_dir, config_path)

    # Run inference and log results to TensorBoard
    run_inference(model, test_images_dir, output_dir, label_map_path, writer)

    # Evaluate model and log metrics to TensorBoard
    evaluate_model(model, test_images_dir, annotation_dir, writer, output_dir)

    # Close TensorBoard writer
    writer.close()
    print(f"TensorBoard logs saved to {log_dir}. You can view them using 'tensorboard --logdir {log_dir}'.")

if __name__ == "__main__":
    main()
