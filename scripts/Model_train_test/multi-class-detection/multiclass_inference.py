import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import sys
import json
import time
import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# --- Configuration from original script ---
# These will be the defaults for the command-line arguments.
DEFAULT_MODEL_PATH = r'F:\SIDG-ATRID-Dataset\flight_demo_data_processed\yolo11m_model\flightdemo_train\weights\best.pt'
DEFAULT_IMAGES_PATH = r'F:\Data collection - real footage\Kraken\Kraken_label.v1i.yolov11\testset\images'
DEFAULT_LABELS_PATH = r'F:\Data collection - real footage\Kraken\Kraken_label.v1i.yolov11\testset\labels'
DEFAULT_OUTPUT_PATH = r'F:\Data collection - real footage\Kraken\Kraken_label.v1i.yolov11\test results\multi-class-sidgatrid-model\inference_video.mp4'

# --- Model and Dataset specific settings ---
# Class names will be loaded dynamically from the model file.
# MODEL_CLASS_NAMES = ['djis900', 'kraken', 'lrquad', 'opterra']


def load_yolo_txt_labels(label_file: Path, image_width: int, image_height: int, ground_truth_class_name: str, model_class_names: List[str]) -> List[Tuple[int, int, int, int, int]]:
    """Load YOLO-format labels and convert to pixel xyxy.
    This function REMAPS the class ID from the label file to the correct ID
    based on the model's class list. This is critical for matching.
    """
    boxes = []
    if not label_file.exists():
        return boxes
        
    # Find the index of the ground truth class in the model's class list
    try:
        # This is the ID the MODEL uses for this class (e.g., 'kraken' might be 1)
        model_gt_class_id = model_class_names.index(ground_truth_class_name)
    except ValueError:
        print(f"[ERROR] Ground truth class '{ground_truth_class_name}' not found in model classes: {model_class_names}.")
        print("[ERROR] Please ensure the '--gt-class-name' or 'data.yaml' matches a name in MODEL_CLASS_NAMES.")
        return [] # Return empty list if class name is invalid

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            # The class_id in the file (e.g., 0) is ignored. We force all GT
            # boxes to use the class ID that the multi-class model expects.
            # This standardizes the labels for correct matching.
            
            xc = float(parts[1]) * image_width
            yc = float(parts[2]) * image_height
            w = float(parts[3]) * image_width
            h = float(parts[4]) * image_height
            x1 = max(0, int(xc - w / 2))
            y1 = max(0, int(yc - h / 2))
            x2 = min(image_width - 1, int(xc + w / 2))
            y2 = min(image_height - 1, int(yc + h / 2))
            boxes.append((model_gt_class_id, x1, y1, x2, y2))
    return boxes


def draw_professional_box(image: np.ndarray, box: Tuple[int, int, int, int], color: Tuple[int, int, int], 
                         alpha: float = 0.3, thickness: int = 3, fill: bool = True):
    """Draw a professional-looking rectangle with optional semi-transparent fill."""
    x1, y1, x2, y2 = box
    
    if fill:
        overlay = image.copy()
        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
        # Blend overlay with original image
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw border
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def put_professional_label(image: np.ndarray, text: str, org: Tuple[int, int], 
                          color: Tuple[int, int, int], bg_color: Tuple[int, int, int] = None):
    """Put text with professional styling including background."""
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    thickness = 1
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    
    if x + text_width > image.shape[1]:
        x = image.shape[1] - text_width - 5
    if y - text_height < 0:
        y = text_height + 5
    
    if bg_color is None:
        bg_color = (0, 0, 0)
    
    padding = 4
    cv2.rectangle(image, 
                 (x - padding, y - text_height - padding), 
                 (x + text_width + padding, y + baseline), 
                 bg_color, -1)
    
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two boxes in xyxy format."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter_area
    return inter_area / union if union > 0 else 0.0


def intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute the intersection area of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    return float(inter_w * inter_h)


def create_excel_analysis(metrics_data: Dict, output_path: Path):
    """Create Excel file with charts and analysis of detection metrics."""
    excel_path = build_output_path(output_path, '_metrics_analysis', '.xlsx')
    
    df_metrics = pd.DataFrame([{
        'Metric': 'True Positives (Correctly Classified)', 'Value': metrics_data['total_tp_correct'], 'Description': 'Correctly detected objects with the correct class'
    }, {
        'Metric': 'True Positives (Misclassified)', 'Value': metrics_data['total_tp_misclassified'], 'Description': 'Correctly detected objects but with the wrong class'
    }, {
        'Metric': 'False Positives', 'Value': metrics_data['total_fp'], 'Description': 'Incorrectly detected objects (ghost detections)'
    }, {
        'Metric': 'False Negatives (Missed)', 'Value': metrics_data['total_fn'], 'Description': 'Ground truth objects that were not detected'
    }, {
        'Metric': 'Precision (Strict)', 'Value': metrics_data['precision_strict'], 'Description': 'TP_correct / (TP_correct + Misclassified + FP)'
    }, {
        'Metric': 'Recall (Strict)', 'Value': metrics_data['recall_strict'], 'Description': 'TP_correct / (TP_correct + FN)'
    }, {
        'Metric': 'Precision (Localization)', 'Value': metrics_data['precision_localization'], 'Description': '(TP_correct + Misclassified) / (TP_correct + Misclassified + FP)'
    }, {
        'Metric': 'Recall (Localization)', 'Value': metrics_data['recall_localization'], 'Description': '(TP_correct + Misclassified) / (TP_correct + Misclassified + FN)'
    }])
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='Metrics Summary', index=False)

    print(f"📊 Created Excel analysis: {excel_path}")
    return excel_path


def create_detailed_results_excel(detailed_data: List[Dict], output_path: Path):
    """Create Excel file with a detailed log of each detection, miss, and false positive."""
    if not detailed_data:
        return None
    
    excel_path = build_output_path(output_path, '_detailed_inference_log', '.xlsx')
    df = pd.DataFrame(detailed_data)
    
    # Define desired column order
    column_order = [
        'frame_number', 'image', 'status', 'gt_class', 'gt_box',
        'predicted_class', 'predicted_box', 'confidence', 'iou'
    ]
    
    # Filter columns to only those that exist in the DataFrame to avoid errors
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    df.to_excel(excel_path, index=False)
    
    print(f"📄 Created detailed inference log: {excel_path}")
    return excel_path


def print_progress(current: int, total: int, prefix: str = "Progress", bar_length: int = 40):
    if total <= 0: return
    ratio = min(max(current / total, 0), 1)
    filled_len = int(bar_length * ratio)
    bar = "#" * filled_len + "-" * (bar_length - filled_len)
    sys.stdout.write(f"\r{prefix}: [{bar}] {current}/{total} ({ratio * 100:.1f}%)")
    sys.stdout.flush()
    if current >= total: sys.stdout.write("\n")


def build_output_path(base_path: Path, suffix: str, extension: str) -> Path:
    if not extension.startswith("."): extension = f".{extension}"
    return base_path.with_name(f"{base_path.stem}{suffix}{extension}")


def load_data_yaml(yaml_path: Path) -> Optional[List[str]]:
    """Safely loads and parses a YAML file to extract class names."""
    try:
        import yaml
    except ImportError:
        print("[ERROR] PyYAML is required to parse data.yaml. Please install it with 'pip install pyyaml'")
        return None
    
    if not yaml_path.exists():
        print(f"[ERROR] data.yaml file not found at: {yaml_path}")
        return None
        
    with open(yaml_path, 'r') as f:
        try:
            data = yaml.safe_load(f)
            if 'names' in data and isinstance(data['names'], list):
                return data['names']
            else:
                print(f"[ERROR] 'names' list not found in {yaml_path}")
        except yaml.YAMLError as e:
            print(f"[ERROR] Error parsing YAML file: {e}")
            return None
    return None


def run(
    images_dir: Path,
    labels_dir: Path,
    model_weights_path: Path,
    output_video_path: Path,
    ground_truth_class_name: str,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.95,
    image_size: int = 640,
    save_per_frame: bool = False,
):
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

    model = YOLO(str(model_weights_path))
    model.conf = confidence_threshold

    # --- Dynamically load model class names ---
    if not hasattr(model, 'names') or not isinstance(model.names, dict) or not model.names:
        print(f"[ERROR] Could not load class names from the model file: {model_weights_path}")
        print("[ERROR] Please ensure the model was trained with class names embedded.")
        return
        
    # model.names is a dict like {0: 'name1', 1: 'name2'}. We need a list in the correct order.
    model_class_names = [model.names[i] for i in sorted(model.names.keys())]
    print(f"[INFO] Model class names loaded: {model_class_names}")


    # --- Class Name and ID Management ---
    # The user provides the GT class *name*. We validate it against the names from the model.
    if ground_truth_class_name not in model_class_names:
        print(f"[ERROR] Ground truth class '{ground_truth_class_name}' is not in the model's class list.")
        print(f"[ERROR] Known model classes are: {model_class_names}")
        return # Exit the run

    try:
        ground_truth_class_id = model_class_names.index(ground_truth_class_name)
    except ValueError:
        # This is redundant due to the check above, but safe to keep.
        print(f"[ERROR] An unexpected error occurred finding '{ground_truth_class_name}' in the model's class list.")
        return 

    print(f"[INFO] Ground truth class for this run: '{ground_truth_class_name}' (ID: {ground_truth_class_id})")

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    print(f"[INFO] Found {len(image_files)} images in '{images_dir}'.")

    first_img = cv2.imread(str(image_files[0]))
    height, width = first_img.shape[:2]
    writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))

    # BGR Colors
    color_gt = (0, 0, 255)         # Red
    color_correct = (0, 255, 0)    # Green
    color_misclassified = (0, 255, 255) # Yellow
    color_fp = (255, 0, 0)         # Blue
    text_color = (255, 255, 255)   # White
    bg_color = (0, 0, 0)           # Black

    frames_dir = None
    if save_per_frame:
        frames_dir = output_video_path.parent / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Saving individual frames to: {frames_dir}")

    # Metrics accumulators
    total_tp_correct = 0
    total_tp_misclassified = 0
    total_fp = 0
    total_fn = 0
    infer_times = []
    detailed_results_data = []

    print_progress(0, len(image_files), prefix="Processing frames")

    for frame_idx, img_path in enumerate(image_files, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Skipping unloadable image: {img_path}")
            continue

        # Ground truth
        lbl_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = load_yolo_txt_labels(lbl_path, width, height, ground_truth_class_name, model_class_names)

        # Prediction
        start = time.perf_counter()
        results = model.predict(source=frame, imgsz=image_size, verbose=False)[0]
        infer_times.append(time.perf_counter() - start)

        pred_boxes_raw = results.boxes.xyxy.cpu().numpy().astype(int)
        pred_scores = results.boxes.conf.cpu().numpy()
        pred_classes = results.boxes.cls.cpu().numpy().astype(int)

        # Match predictions to GT boxes
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        # This list will hold tuples of (gt_index, pred_index, iou)
        matches = []

        if len(gt_boxes) > 0 and len(pred_boxes_raw) > 0:
            # This loop tries to find the best prediction for each ground truth
            for i, (gt_cls_id_from_box, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                # Sanity check: the ID from the box should match the expected GT ID for this run.
                if gt_cls_id_from_box != ground_truth_class_id:
                    print(f"[WARN] Frame {frame_idx}: Mismatched GT class ID! Expected {ground_truth_class_id} but found {gt_cls_id_from_box}.")
                    continue # Skip this GT box as it's unexpected

                gt_box = (gx1, gy1, gx2, gy2)
                best_iou = 0.0
                best_pred_idx = -1
                
                for j, pbox in enumerate(pred_boxes_raw):
                    iou = iou_xyxy(gt_box, pbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = j
                
                # Check if this best-IoU prediction is a valid match
                is_a_match = False
                if best_iou > iou_threshold:
                    is_a_match = True
                else:
                    # Fallback check: if IoU is low, check for containment
                    # This helps when GT boxes are too large / predictions are "too good"
                    if best_pred_idx != -1:
                        pred_box = pred_boxes_raw[best_pred_idx]
                        inter = intersection_area(gt_box, pred_box)
                        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                        containment = inter / pred_area if pred_area > 0 else 0
                        
                        if containment > containment_threshold:
                            is_a_match = True

                if is_a_match:
                    # Avoid matching a prediction to multiple GTs
                    if best_pred_idx not in matched_pred_indices:
                        # Use the actual IoU for logging, even if containment was the reason for matching
                        matches.append((i, best_pred_idx, best_iou))
                        matched_gt_indices.add(i)
                        matched_pred_indices.add(best_pred_idx)


        # --- Frame-level analysis ---
        frame_tp_correct = 0
        frame_tp_misclassified = 0
        frame_fp = 0
        frame_fn = 0
        
        # Draw GT boxes
        for i, (gt_cls_id, x1, y1, x2, y2) in enumerate(gt_boxes):
            is_missed = i not in matched_gt_indices
            gt_color = (0, 0, 128) if is_missed else color_gt # Darker red if missed
            draw_professional_box(frame, (x1, y1, x2, y2), gt_color, fill=False, thickness=2)
            label_text = f"GT: {model_class_names[gt_cls_id]}"
            if is_missed:
                label_text += " (MISSED)"
            put_professional_label(frame, label_text, (x1, max(25, y1 - 10)), text_color, bg_color)

        # Process matches (TPs - correct or misclassified)
        for gt_idx, pred_idx, iou in matches:
            gt_cls_id, gx1, gy1, gx2, gy2 = gt_boxes[gt_idx]
            pred_cls_id = pred_classes[pred_idx]
            
            pbox = pred_boxes_raw[pred_idx]
            px1, py1, px2, py2 = pbox
            score = pred_scores[pred_idx]
            
            if pred_cls_id == gt_cls_id:
                status = 'Correctly Classified (TP)'
                total_tp_correct += 1
                frame_tp_correct += 1
                color = color_correct
                label_text = f"Pred: {model_class_names[pred_cls_id]} ({score:.2f})"
            else:
                status = 'Misclassified (TP)'
                total_tp_misclassified += 1
                frame_tp_misclassified += 1
                color = color_misclassified
                label_text = f"Pred: {model_class_names[pred_cls_id]} ({score:.2f}) -> GT: {model_class_names[gt_cls_id]}"

            detailed_results_data.append({
                'image': img_path.name, 'frame_number': frame_idx, 'status': status,
                'gt_class': model_class_names[gt_cls_id], 'gt_box': [gx1, gy1, gx2, gy2],
                'predicted_class': model_class_names[pred_cls_id], 'predicted_box': pbox.tolist(),
                'confidence': score, 'iou': iou
            })

            draw_professional_box(frame, (px1, py1, px2, py2), color, fill=True, alpha=0.15, thickness=2)
            put_professional_label(frame, label_text, (px1, max(25, py2 + 15)), text_color, bg_color)

        # Process FPs (unmatched predictions)
        for i in range(len(pred_boxes_raw)):
            if i not in matched_pred_indices:
                total_fp += 1
                frame_fp += 1
                pbox = pred_boxes_raw[i]
                px1, py1, px2, py2 = pbox
                score = pred_scores[i]
                pred_cls_id = pred_classes[i]

                detailed_results_data.append({
                    'image': img_path.name, 'frame_number': frame_idx, 'status': 'False Positive (FP)',
                    'gt_class': None, 'gt_box': None,
                    'predicted_class': model_class_names[pred_cls_id], 'predicted_box': pbox.tolist(),
                    'confidence': score, 'iou': None
                })

                label_text = f"FP: {model_class_names[pred_cls_id]} ({score:.2f})"
                draw_professional_box(frame, (px1, py1, px2, py2), color_fp, fill=True, alpha=0.15, thickness=2)
                put_professional_label(frame, label_text, (px1, max(25, py2 + 15)), text_color, bg_color)
        
        # Process FNs (unmatched GTs)
        for i, (gt_cls_id, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            if i not in matched_gt_indices:
                detailed_results_data.append({
                    'image': img_path.name, 'frame_number': frame_idx, 'status': 'Miss (FN)',
                    'gt_class': model_class_names[gt_cls_id], 'gt_box': [gx1, gy1, gx2, gy2],
                    'predicted_class': None, 'predicted_box': None, 'confidence': None, 'iou': None
                })

        frame_fn = len(gt_boxes) - len(matched_gt_indices)
        total_fn += frame_fn
        
        # Frame info
        frame_info = f"Frame: {frame_idx}/{len(image_files)} | Correct: {frame_tp_correct} | Misclassified: {frame_tp_misclassified} | FP: {frame_fp} | Missed: {frame_fn}"
        put_professional_label(frame, frame_info, (10, 30), text_color, bg_color)
        
        if save_per_frame:
            if frames_dir:
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), frame)

        writer.write(frame)
        print_progress(frame_idx, len(image_files), prefix="Processing frames")

    writer.release()
    
    # Final metrics
    precision_strict = total_tp_correct / (total_tp_correct + total_tp_misclassified + total_fp) if (total_tp_correct + total_tp_misclassified + total_fp) > 0 else 0
    recall_strict = total_tp_correct / (total_tp_correct + total_fn) if (total_tp_correct + total_fn) > 0 else 0
    
    precision_localization = (total_tp_correct + total_tp_misclassified) / (total_tp_correct + total_tp_misclassified + total_fp) if (total_tp_correct + total_tp_misclassified + total_fp) > 0 else 0
    recall_localization = (total_tp_correct + total_tp_misclassified) / (total_tp_correct + total_tp_misclassified + total_fn) if (total_tp_correct + total_tp_misclassified + total_fn) > 0 else 0
    
    metrics_data = {
        'total_tp_correct': total_tp_correct, 'total_tp_misclassified': total_tp_misclassified,
        'total_fp': total_fp, 'total_fn': total_fn,
        'precision_strict': precision_strict, 'recall_strict': recall_strict,
        'precision_localization': precision_localization, 'recall_localization': recall_localization
    }
    
    create_excel_analysis(metrics_data, output_video_path)
    if save_per_frame:
        create_detailed_results_excel(detailed_results_data, output_video_path)

    print(f"\n🎬 Saved annotated video to: {output_video_path}")
    print("\n📊 Metrics Summary:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ Correctly Classified (TP): {total_tp_correct}")
    print(f"⚠️  Misclassified (TP):      {total_tp_misclassified}")
    print(f"❌ False Positives (FP):      {total_fp}")
    print(f"⭕ False Negatives (FN):      {total_fn}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🎯 Precision (Strict): {precision_strict:.4f}")
    print(f"🔍 Recall (Strict):    {recall_strict:.4f}")
    print(f"--------------------------------------------------------")
    print(f"🎯 Precision (Localization): {precision_localization:.4f}")
    print(f"🔍 Recall (Localization):    {recall_localization:.4f}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiclass model inference on a single-class test set and generate detailed analysis.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_PATH, help="Directory containing test images")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_PATH, help="Directory containing YOLO label files")
    parser.add_argument("--model-weights", type=Path, default=DEFAULT_MODEL_PATH, help="Path to model weights file (best.pt)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output MP4 video path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.25, help="IoU threshold for matching")
    parser.add_argument("--containment-threshold", type=float, default=0.95, help="Fallback threshold for containment if IoU fails")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    parser.add_argument("--per-frame-img", action="store_true", help="Save each processed frame as an image and create per-frame Excel")
    parser.add_argument("--data-yaml", type=Path, default=None, help="Path to the data.yaml file to automatically determine the ground truth class name.")
    parser.add_argument("--gt-class-name", type=str, default=None, help="Manually specify the ground truth class name if no data.yaml is available.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --- Determine Ground Truth Class Name ---
    gt_class_name = None
    # We can't validate the class name until we load the model, so we just get the name here.
    if args.data_yaml:
        names = load_data_yaml(args.data_yaml)
        if names:
            gt_class_name = names[0] # Assume the first class is the ground truth
            print(f"[INFO] Using ground truth class '{gt_class_name}' from {args.data_yaml}")
        else:
            sys.exit(1) # Exit if YAML parsing fails
    elif args.gt_class_name:
        gt_class_name = args.gt_class_name
        print(f"[INFO] Using manually specified ground truth class: '{gt_class_name}'")
    
    if not gt_class_name:
        print("[ERROR] You must specify the ground truth class.")
        print("Please provide the dataset's configuration file using '--data-yaml' or specify the class name directly using '--gt-class-name'.")
        sys.exit(1)

    run(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        model_weights_path=args.model_weights,
        output_video_path=args.output,
        ground_truth_class_name=gt_class_name,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        containment_threshold=args.containment_threshold,
        image_size=args.imgsz,
        save_per_frame=args.per_frame_img,
    )
