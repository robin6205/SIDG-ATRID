import os
import json
import cv2
import numpy as np
from ultralytics import YOLO # Assumes YOLOv8 or a compatible version
from pathlib import Path

# --- Configuration ---
BASE_INPUT_PATH = r"D:\\SiDG-ATRID-Dataset\\Train_set\\sidg-atrid-dataset-urban-processed\\DJIS900\\brushify-urban"
MODEL_PATH = r"D:\SiDG-ATRID-Dataset\Model_test_results\Models\warsaw_dataset_model-yolo11\warsaw_dataset_model\weights\\best.pt"
BASE_OUTPUT_PATH = r"D:\SiDG-ATRID-Dataset\Train_set\sidg-atrid-dataset-urban-processed\\sensor-placement"
CAMERA_FOLDERS = ["urban-clearday-cam1", "urban-clearday-cam2", "urban-clearday-cam3", "urban-clearday-cam4", "urban-cloudy-cam1", "urban-cloudy-cam2", "urban-cloudy-cam3", "urban-cloudy-cam4", "urban-foggy-cam1", "urban-foggy-cam2", "urban-foggy-cam3", "urban-foggy-cam4", "urban-rainy-cam1", "urban-rainy-cam2", "urban-rainy-cam3", "urban-rainy-cam4"]

IOU_THRESHOLD = 0.25  # Tunable IoU threshold for a detection to be considered successful
CONFIDENCE_THRESHOLD = 0.25  # Confidence threshold for YOLO predictions
DRONE_CLASS_ID = 0  # Assuming drone is class 0 in the YOLO model

# --- Helper Functions ---

def get_frame_index_from_filename(filename):
    """Extracts the leading integer (frame index) from a filename."""
    try:
        return int(filename.split('_')[0])
    except (ValueError, IndexError):
        return -1 # Fallback for unexpected filename formats, will sort them first

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

# --- Main Processing Logic ---
def main():
    model = load_yolo_model(MODEL_PATH)

    for camera_folder_name in CAMERA_FOLDERS:
        print(f"\\nProcessing camera folder: {camera_folder_name}")
        
        input_camera_path = os.path.join(BASE_INPUT_PATH, camera_folder_name)
        output_camera_path = os.path.join(BASE_OUTPUT_PATH, camera_folder_name)
        
        images_dir = os.path.join(input_camera_path, "images")
        labels_dir = os.path.join(input_camera_path, "labels")
        
        output_debug_images_dir = os.path.join(output_camera_path, "debug-images")
        os.makedirs(output_camera_path, exist_ok=True) # Ensure camera output base exists
        os.makedirs(output_debug_images_dir, exist_ok=True) # Ensure debug images dir exists
        
        successful_detection_filenames = []
        successful_detection_indices = []
        detection_status_all_images = []

        if not os.path.exists(images_dir):
            print(f"  Warning: Images directory not found: {images_dir}")
            continue

        # Sort numerically by the leading frame index in the filename
        all_files_in_dir = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files = sorted(all_files_in_dir, key=get_frame_index_from_filename)
        
        for idx, image_file_name in enumerate(image_files): # idx is now the 0-based index of the correctly sorted list
            image_path = os.path.join(images_dir, image_file_name)
            
            image_stem = Path(image_file_name).stem
            if image_stem.endswith('_lit'):
                label_base_name = image_stem[:-4]
            else:
                label_base_name = image_stem
            
            label_file_name = f"{label_base_name}_lit.txt"
            label_path = os.path.join(labels_dir, label_file_name)

            print(f"  Processing image: {image_file_name} (expecting label: {label_file_name})")

            img = cv2.imread(image_path)
            if img is None:
                print(f"    Warning: Could not read image {image_path}")
                continue
            
            img_height, img_width = img.shape[:2]

            gt_boxes = parse_yolo_label_file(label_path, img_width, img_height)

            results = model(img, conf=CONFIDENCE_THRESHOLD, classes=[DRONE_CLASS_ID], verbose=False)
            
            pred_boxes = []
            if results and results[0].boxes:
                for box_data in results[0].boxes:
                    # Already filtered by class and conf by model parameters
                    xyxy = box_data.xyxy[0].tolist()
                    pred_boxes.append(xyxy)
            
            # --- Matching Logic ---
            # gt_status: True if detected, best_iou, matching_pred_box
            gt_match_status = [{'detected': False, 'iou': 0.0} for _ in gt_boxes]
            # pred_status: True if TP, best_iou_with_gt, matched_gt_box
            pred_match_status = [{'is_tp': False, 'iou': 0.0} for _ in pred_boxes]

            # Use a copy of GT indices to ensure each GT is matched at most once for TP assignment
            # This helps in correctly identifying TPs vs FPs
            available_gt_indices = list(range(len(gt_boxes)))
            
            # Sort predictions, e.g., by area or confidence if needed, though not strictly necessary for this logic
            # For each prediction, try to match it to an available GT box
            for i, pred_b in enumerate(pred_boxes):
                best_iou_for_this_pred = -1
                best_gt_idx_for_this_pred = -1 # Index in original gt_boxes
                best_available_gt_local_idx = -1 # Index in available_gt_indices

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
                    
                    # Mark the corresponding GT as detected and store its best match IoU
                    if not gt_match_status[best_gt_idx_for_this_pred]['detected'] or \
                       best_iou_for_this_pred > gt_match_status[best_gt_idx_for_this_pred]['iou']:
                        gt_match_status[best_gt_idx_for_this_pred]['detected'] = True
                        gt_match_status[best_gt_idx_for_this_pred]['iou'] = best_iou_for_this_pred
                    
                    # Remove this GT from being available for other predictions
                    available_gt_indices.pop(best_available_gt_local_idx)


            image_has_any_successful_detection = any(gs['detected'] for gs in gt_match_status)

            # Determine if a debug image should be saved
            save_debug_this_image = False
            debug_image_suffix = "_debug.jpg"

            if image_has_any_successful_detection:
                successful_detection_filenames.append(image_file_name)
                # Store the actual frame index from the filename
                parsed_frame_index = get_frame_index_from_filename(image_file_name)
                if parsed_frame_index != -1: # Ensure parsing was successful
                    successful_detection_indices.append(parsed_frame_index)
                else:
                    print(f"    Warning: Could not parse frame index from filename {image_file_name} for successful_detection_indices.")
                    successful_detection_indices.append(idx) # Fallback to sorted index if parsing fails
                
                detection_status_all_images.append(1)
                print(f"    SUCCESS: Drone detected in {image_file_name}.")
                save_debug_this_image = True
            elif gt_boxes and not image_has_any_successful_detection: # False Negatives exist
                detection_status_all_images.append(0)
                print(f"    INFO: No drone detected in {image_file_name} (had GT, all missed).")
                save_debug_this_image = True # Save to show FNs
                debug_image_suffix = "_FN.jpg"
            elif not gt_boxes and pred_boxes: # False Positives exist, no GT
                detection_status_all_images.append(0) # No GT, so not a successful detection of a GT drone
                print(f"    INFO: False positive(s) in {image_file_name} (no GT drone).")
                save_debug_this_image = True # Save to show FPs
                debug_image_suffix = "_FP.jpg"
            else: # No GT and No Preds
                detection_status_all_images.append(0)

            if save_debug_this_image:
                debug_img = img.copy()
                # Draw Ground Truth Boxes
                for j, gt_b in enumerate(gt_boxes):
                    status = gt_match_status[j]
                    if status['detected']:
                        color = (0, 255, 0) # Green
                        label = f"GT (Detected IoU:{status['iou']:.2f})"
                    else:
                        color = (0, 165, 255) # Orange
                        label = "GT (Missed)"
                    draw_bounding_box(debug_img, gt_b, color, label)
                
                # Draw Predicted Boxes
                for i, pred_b in enumerate(pred_boxes):
                    status = pred_match_status[i]
                    if status['is_tp']:
                        color = (255, 0, 0) # Blue
                        label = f"Pred (TP IoU:{status['iou']:.2f})"
                    else:
                        color = (0, 0, 255) # Red
                        label = "Pred (FP)"
                    draw_bounding_box(debug_img, pred_b, color, label)
                
                debug_image_filename = f"{Path(image_file_name).stem}{debug_image_suffix}"
                debug_image_path = os.path.join(output_debug_images_dir, debug_image_filename)
                try:
                    cv2.imwrite(debug_image_path, debug_img)
                    print(f"      Saved debug image: {debug_image_path}")
                except Exception as e:
                    print(f"      Error saving debug image {debug_image_path}: {e}")
        
        output_data = {
            "successful_detection_filenames": successful_detection_filenames,
            "successful_detection_indices": successful_detection_indices,
            "detection_status_all_images": detection_status_all_images
        }
        
        output_json_path = os.path.join(output_camera_path, f"{camera_folder_name}_detection_summary.json") # Renamed for clarity
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"  Saved detection summary to {output_json_path}")

if __name__ == "__main__":
    main() 