import torch
import os
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
from ultralytics import YOLO
import pandas as pd
import shutil

def load_model(weights_path, conf_threshold=0.25):
    """Load YOLO model with specified confidence threshold"""
    model = YOLO(weights_path)
    model.conf = conf_threshold  # Set confidence threshold
    return model

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    # Add debug prints
    print(f"\nIoU Calculation Debug:")
    print(f"Box1: {box1}")
    print(f"Box2: {box2}")
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    iou = intersection / union if union > 0 else 0
    
    print(f"Intersection: {intersection}")
    print(f"Union: {union}")
    print(f"IoU: {iou}")
    
    return iou

def preprocess_image(img, img_size=640):
    """Preprocess image consistently with training pipeline"""
    # Resize image maintaining aspect ratio
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create square image with black padding
    square_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    offset_h = (img_size - new_h) // 2
    offset_w = (img_size - new_w) // 2
    square_img[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = resized
    
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)
    
    return rgb_img, (scale, offset_h, offset_w)

def evaluate_model(model, source_path, processed_csv_dir, output_path, img_size=640, iou_threshold=0.5):
    """
    Evaluate model using original images but processed CSV files
    
    Args:
        model: YOLO model
        source_path: Path to original images
        processed_csv_dir: Path to processed CSV files
        output_path: Path to save results
        img_size: Image size for processing (default: 640)
        iou_threshold: IoU threshold for true positive detection (default: 0.5)
    """
    # drone_types = ['DJIPhantom','Anafi-Extended', 'DJIFPV', 'Mavic_Air', 'Mavic_Enterprise', 'EFT-E410S']
    drone_types = ['EFT-E410S']
    weather_conditions = ['cloudy', 'evening', 'sunny']
    
    results = {
        'total_images': 0,
        'total_detections': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'weather_wise_metrics': {},
        'folder_wise_metrics': {},  # New dictionary for detailed breakdown
        'inference_times': []
    }
    
    # Initialize folder-wise metrics
    for drone_type in drone_types:
        results['folder_wise_metrics'][drone_type] = {}
        for condition in weather_conditions:
            results['folder_wise_metrics'][drone_type][condition] = {
                'images': 0,
                'detections': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'avg_inference_time': 0,
                'inference_times': []
            }

    base_path = Path(source_path)
    output_path = Path(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Create debug directory for raw predictions
    debug_dir = output_path / 'debug'
    os.makedirs(debug_dir, exist_ok=True)
    
    # Add debug logging for IoU calculations
    def log_detection_debug(img_filename, gt_box, pred_box, iou, condition, drone_type):
        print(f"\nDebug for {drone_type}/{condition}/{img_filename}")
        print(f"Ground Truth Box: {gt_box}")
        print(f"Prediction Box: {pred_box}")
        print(f"IoU: {iou:.4f}")
        print(f"IoU Threshold Check: {iou >= iou_threshold}")

    for drone_type in drone_types:
        print(f"\nEvaluating drone type: {drone_type}")
        drone_vis_path = output_path / drone_type
        os.makedirs(drone_vis_path, exist_ok=True)
        
        for condition in weather_conditions:
            print(f"\nProcessing {drone_type}-{condition}")
            condition_vis_path = drone_vis_path / condition
            os.makedirs(condition_vis_path, exist_ok=True)
            
            # Reset folder-specific detection counter
            folder_total_detections = 0
            image_count = 0
            
            processed_csv_path = Path(processed_csv_dir) / drone_type / 'labels' / f'{condition}_processed.csv'
            if not processed_csv_path.exists():
                print(f"WARNING: Processed CSV file not found: {processed_csv_path}")
                continue
            
            try:
                df = pd.read_csv(processed_csv_path)
                print(f"Found {len(df)} annotations in {processed_csv_path}")
                
                # Ensure unique image entries
                df = df.drop_duplicates(subset=['image_file'])
                
                if condition not in results['weather_wise_metrics']:
                    results['weather_wise_metrics'][condition] = {
                        'images': 0,
                        'detections': 0,
                        'true_positives': 0,
                        'false_positives': 0,
                        'false_negatives': 0
                    }
                
                # Limit processing to the first 20 images
                for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
                    if idx >= 20:
                        break
                    
                    try:
                        img_filename = row['image_file']
                        image_count += 1
                        
                        img_path = Path(source_path) / drone_type / 'images' / condition.capitalize() / img_filename
                        if not img_path.exists():
                            print(f"Image not found: {img_path}")
                            continue
                        
                        # Read and preprocess image
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                            
                        # Print image info for first few images
                        if results['total_images'] < 5:
                            print(f"\nImage info for: {img_path}")
                            print(f"Shape: {img.shape}")
                            print(f"Range: {img.min()} - {img.max()}")
                            print(f"Mean: {img.mean():.2f}")
                            print(f"Std: {img.std():.2f}")
                        
                        # Preprocess image
                        processed_img, (scale, offset_h, offset_w) = preprocess_image(img, img_size)
                        
                        # Get ground truth box
                        gt_box = [
                            float(row['xmin']),
                            float(row['ymin']),
                            float(row['xmax']),
                            float(row['ymax'])
                        ]
                        
                        # Model inference
                        start_time = datetime.now()
                        results_pred = model(processed_img)[0]
                        inference_time = (datetime.now() - start_time).total_seconds()
                        results['inference_times'].append(inference_time)
                        
                        # Create visualization
                        vis_img = img.copy()
                        
                        # Draw ground truth box (green)
                        cv2.rectangle(vis_img, 
                                    (int(gt_box[0]), int(gt_box[1])),
                                    (int(gt_box[2]), int(gt_box[3])),
                                    (0, 255, 0), 2)
                        
                        # Process predictions
                        if len(results_pred.boxes) == 0:
                            print(f"Image {image_count} - 0 detections - folder total: {folder_total_detections}")
                            results['false_negatives'] += 1
                            results['weather_wise_metrics'][condition]['false_negatives'] += 1
                            results['folder_wise_metrics'][drone_type][condition]['false_negatives'] += 1
                        else:
                            num_detections = len(results_pred.boxes)
                            folder_total_detections += num_detections
                            
                            print(f"Image {image_count} - {num_detections} detection(s) - folder total: {folder_total_detections}")
                            print(f"Current folder: {drone_type}/{condition}")
                            
                            # Track which predictions have been matched
                            matched_predictions = set()
                            best_iou = 0
                            best_pred_idx = None
                            
                            # Process each prediction
                            for pred_idx, pred_box in enumerate(results_pred.boxes):
                                box = pred_box.xyxy[0].cpu().numpy()
                                conf = float(pred_box.conf.cpu().numpy()[0])
                                
                                # Adjust coordinates for padding and scaling
                                box_orig = box.copy()
                                box_orig[0] = (box[0] - offset_w) / scale
                                box_orig[1] = (box[1] - offset_h) / scale
                                box_orig[2] = (box[2] - offset_w) / scale
                                box_orig[3] = (box[3] - offset_h) / scale
                                
                                iou = calculate_iou(gt_box, box_orig)
                                
                                # Track best prediction for this ground truth
                                if iou > best_iou:
                                    best_iou = iou
                                    best_pred_idx = pred_idx
                                
                                # Draw prediction box (blue for TP, red for FP)
                                color = (255, 0, 0) if iou >= iou_threshold else (0, 0, 255)
                                cv2.rectangle(vis_img,
                                            (int(box_orig[0]), int(box_orig[1])),
                                            (int(box_orig[2]), int(box_orig[3])),
                                            color, 2)
                                cv2.putText(vis_img,
                                          f'{conf:.2f}',
                                          (int(box_orig[0]), int(box_orig[1] - 5)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Update metrics based on best match
                            if best_iou >= iou_threshold and best_pred_idx is not None:
                                # True positive case
                                matched_predictions.add(best_pred_idx)
                                results['true_positives'] += 1
                                results['weather_wise_metrics'][condition]['true_positives'] += 1
                                results['folder_wise_metrics'][drone_type][condition]['true_positives'] += 1
                                print("Found true positive detection")
                            else:
                                # False negative case
                                results['false_negatives'] += 1
                                results['weather_wise_metrics'][condition]['false_negatives'] += 1
                                results['folder_wise_metrics'][drone_type][condition]['false_negatives'] += 1
                                print("No valid match - counted as false negative")
                            
                            # Count unmatched predictions as false positives
                            unmatched_count = num_detections - len(matched_predictions)
                            if unmatched_count > 0:
                                results['false_positives'] += unmatched_count
                                results['weather_wise_metrics'][condition]['false_positives'] += unmatched_count
                                results['folder_wise_metrics'][drone_type][condition]['false_positives'] += unmatched_count
                                print(f"Found {unmatched_count} false positive detections")
                            
                            # Update detection counts
                            results['total_detections'] += num_detections
                            results['weather_wise_metrics'][condition]['detections'] += num_detections
                            results['folder_wise_metrics'][drone_type][condition]['detections'] = folder_total_detections
                        
                        # Save visualization
                        vis_path = condition_vis_path / img_filename
                        cv2.imwrite(str(vis_path), vis_img)
                        
                        # Save debug image with raw predictions
                        debug_path = debug_dir / f"{drone_type}_{condition}_{img_filename}"
                        cv2.imwrite(str(debug_path), processed_img[..., ::-1])  # Convert back to BGR
                        
                        results['total_images'] += 1
                        results['weather_wise_metrics'][condition]['images'] += 1
                        results['folder_wise_metrics'][drone_type][condition]['images'] += 1
                        
                        results['folder_wise_metrics'][drone_type][condition]['inference_times'].append(inference_time)
                        
                    except Exception as e:
                        print(f"Error processing image {img_filename}: {str(e)}")
                        continue
                
                print(f"\nFinished {drone_type}/{condition}")
                print(f"Total images processed: {image_count}")
                print(f"Total detections in folder: {folder_total_detections}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error processing condition {condition}: {str(e)}")
                continue
    
    # Calculate metrics
    if results['total_images'] > 0:
        # Calculate overall metrics
        tp = results['true_positives']
        fp = results['false_positives']
        fn = results['false_negatives']
        
        results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results['f1_score'] = 2 * (results['precision'] * results['recall']) / \
            (results['precision'] + results['recall']) if (results['precision'] + results['recall']) > 0 else 0
        results['avg_inference_time'] = sum(results['inference_times']) / len(results['inference_times'])
        
        # Calculate weather-wise metrics
        for condition, metrics in results['weather_wise_metrics'].items():
            if metrics['images'] > 0:
                tp = metrics['true_positives']
                fp = metrics['false_positives']
                fn = metrics['false_negatives']
                
                metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                    (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # Calculate folder-wise metrics
        for drone_type in results['folder_wise_metrics']:
            for condition, metrics in results['folder_wise_metrics'][drone_type].items():
                if metrics['images'] > 0:
                    tp = metrics['true_positives']
                    fp = metrics['false_positives']
                    fn = metrics['false_negatives']
                    
                    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                        (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
                    metrics['avg_inference_time'] = sum(metrics['inference_times']) / len(metrics['inference_times']) if metrics['inference_times'] else 0
    
    # Save results as JSON
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create and save detailed CSV report
    csv_data = []
    for drone_type in results['folder_wise_metrics']:
        for condition, metrics in results['folder_wise_metrics'][drone_type].items():
            row = {
                'drone_type': drone_type,
                'weather_condition': condition,
                'total_images': metrics['images'],
                'total_detections': metrics['detections'],
                'true_positives': metrics['true_positives'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'avg_inference_time': metrics['avg_inference_time']
            }
            csv_data.append(row)
    
    # Save detailed CSV report
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path / 'detailed_results.csv', index=False)
    
    return results

def preprocess_visiodect_dataset(source_path, output_path):
    """
    Preprocess VisioDECT dataset CSV/XLSX files with corrected bounding box calculations
    
    Args:
        source_path: Path to original VisioDECT dataset
        output_path: Path to save processed CSV files
    """
    drone_types = ['DJIPhantom','Anafi-Extended', 'DJIFPV', 'Mavic_Air', 'Mavic_Enterprise', 'EFT-E410S']
    weather_conditions = ['cloudy', 'evening', 'sunny']
    
    source_path = Path(source_path)
    output_path = Path(output_path)
    
    processed_summary = {drone: {cond: 0 for cond in weather_conditions} for drone in drone_types}
    
    print("\nStarting VisioDECT dataset preprocessing...")
    print(f"Source path: {source_path}")
    print(f"Output path: {output_path}")
    
    # Create output directory structure
    for drone_type in drone_types:
        drone_dir = output_path / drone_type / 'labels'
        os.makedirs(drone_dir, exist_ok=True)
        
        print(f"\nProcessing drone type: {drone_type}")
        for condition in weather_conditions:
            print(f"\nProcessing {drone_type}-{condition}")
            
            # Try both CSV and XLSX files
            csv_path = source_path / drone_type / 'labels' / condition / 'csv.csv'
            xlsx_path = source_path / drone_type / 'labels' / condition / 'csv.xlsx'
            
            if csv_path.exists():
                input_path = csv_path
                is_xlsx = False
            elif xlsx_path.exists():
                input_path = xlsx_path
                is_xlsx = True
            else:
                print(f"WARNING: Neither CSV nor XLSX file found in: {source_path / drone_type / 'labels' / condition}")
                continue
            
            # Verify images directory exists
            img_dir = source_path / drone_type / 'images' / condition.capitalize()
            if not img_dir.exists():
                print(f"WARNING: Image directory not found: {img_dir}")
                continue
            
            try:
                # Read input file based on format
                if is_xlsx:
                    df = pd.read_excel(input_path, header=None, names=[
                        'class_name', 'x', 'y', 'w', 'h', 'image_file', 'img_width', 'img_height'
                    ])
                else:
                    df = pd.read_csv(input_path, header=None, names=[
                        'class_name', 'x', 'y', 'w', 'h', 'image_file', 'img_width', 'img_height'
                    ])
                
                print(f"Found {len(df)} entries in {input_path}")
                
                # Create new DataFrame for processed data
                processed_data = []
                
                for _, row in tqdm(df.iterrows(), desc=f"Processing {drone_type}-{condition}"):
                    try:
                        # Verify image exists and get actual dimensions
                        img_path = img_dir / row['image_file']
                        if not img_path.exists():
                            print(f"Image not found: {img_path}")
                            continue
                        
                        img = cv2.imread(str(img_path))
                        if img is None:
                            print(f"Failed to read image: {img_path}")
                            continue
                        
                        actual_height, actual_width = img.shape[:2]
                        
                        # Get original coordinates and dimensions
                        x = float(row['x'])
                        y = float(row['y'])
                        width = float(row['w'])
                        height = float(row['h'])
                        
                        # Calculate box corners (ensure within image bounds)
                        xmin = max(0, x)  # Original x is already the left edge
                        ymin = max(0, y)  # Original y is already the top edge
                        xmax = min(actual_width, x + width)
                        ymax = min(actual_height, y + height)
                        
                        # Calculate center coordinates
                        center_x = x + width/2
                        center_y = y + height/2
                        
                        # Store processed data
                        processed_data.append({
                            'image_file': row['image_file'],
                            'img_width': actual_width,
                            'img_height': actual_height,
                            'xmin': xmin,
                            'ymin': ymin,
                            'xmax': xmax,
                            'ymax': ymax,
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': width,
                            'height': height
                        })
                        
                    except Exception as e:
                        print(f"Error processing row: {row}")
                        print(f"Error: {str(e)}")
                        continue
                
                # Save processed data
                output_csv_path = drone_dir / f'{condition}_processed.csv'
                processed_df = pd.DataFrame(processed_data)
                processed_df.to_csv(output_csv_path, index=False)
                
                processed_summary[drone_type][condition] = len(processed_data)
                print(f"Successfully processed {len(processed_data)} entries for {drone_type}-{condition}")
                print(f"Saved to {output_csv_path}")
                
            except Exception as e:
                print(f"Error processing file {input_path}: {str(e)}")
                continue
    
    return processed_summary

if __name__ == "__main__":
    # Paths
    source_path = r"D:\SiDG-ATRID-Dataset\VisioDECT Dataset\VisioDECT Dataset Upload"
    processed_data_path = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\test_set\visio_dect_yolo_testset"
    model_path = r"D:\SiDG-ATRID-Dataset\sidg-atrid-model\model\train\weights\best.pt"
    output_path = r"D:\SiDG-ATRID-Dataset\sidg-atrid-model\visiodect_dataset_test"

    # Threshold settings
    CONFIDENCE_THRESHOLD = 0.6  # Model confidence threshold
    IOU_THRESHOLD = 0.2        # IoU threshold for true positive detection
    IMG_SIZE = 640            # Image size for processing

    # Step 1: Preprocess the VisioDECT dataset
    # print("Step 1: Preprocessing VisioDECT dataset...")
    # preprocessing_summary = preprocess_visiodect_dataset(source_path, processed_data_path)
    
    # # Print preprocessing summary
    # print("\nPreprocessing Summary:")
    # print("=====================")
    # for drone in preprocessing_summary:
    #     print(f"\n{drone}:")
    #     for condition in preprocessing_summary[drone]:
    #         count = preprocessing_summary[drone][condition]
    #         print(f"  {condition}: {count} images processed")

    # Step 2: Load model and evaluate
    print("\nStep 2: Evaluating model...")
    model = load_model(model_path, conf_threshold=CONFIDENCE_THRESHOLD)
    results = evaluate_model(model, source_path, processed_data_path, output_path, 
                           img_size=IMG_SIZE, iou_threshold=IOU_THRESHOLD)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Total Images: {results['total_images']}")
    print(f"Total Detections: {results['total_detections']}")
    print(f"True Positives: {results['true_positives']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"False Negatives: {results['false_negatives']}")
    
    if 'precision' in results:
        print(f"\nOverall Metrics:")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Average Inference Time: {results['avg_inference_time']:.4f} seconds")
    
    print("\nWeather-wise Results:")
    for condition, metrics in results['weather_wise_metrics'].items():
        if metrics['images'] > 0:
            print(f"\n{condition.upper()}:")
            print(f"Images: {metrics['images']}")
            print(f"Detections: {metrics['detections']}")
            if 'precision' in metrics:
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1_score']:.4f}")

    # Print detailed results
    print("\nDetailed Results by Folder:")
    print("==========================")
    for drone_type in results['folder_wise_metrics']:
        print(f"\n{drone_type}:")
        for condition, metrics in results['folder_wise_metrics'][drone_type].items():
            if metrics['images'] > 0:
                print(f"\n  {condition.upper()}:")
                print(f"    Images: {metrics['images']}")
                print(f"    Detections: {metrics['detections']}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    F1 Score: {metrics['f1_score']:.4f}")
                print(f"    Avg Inference Time: {metrics['avg_inference_time']:.4f} seconds") 