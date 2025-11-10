import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict
import time
import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO


def load_yolo_txt_labels(label_file: Path, image_width: int, image_height: int) -> List[Tuple[int, float, float, float, float]]:
	"""Load YOLO-format labels and convert to pixel xyxy.

	Returns list of (class_id, x1, y1, x2, y2) in pixels.
	"""
	boxes = []
	if not label_file.exists():
		return boxes
	with open(label_file, "r") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) != 5:
				continue
			cls_id = int(float(parts[0]))
			xc = float(parts[1]) * image_width
			yc = float(parts[2]) * image_height
			w = float(parts[3]) * image_width
			h = float(parts[4]) * image_height
			x1 = max(0, int(xc - w / 2))
			y1 = max(0, int(yc - h / 2))
			x2 = min(image_width - 1, int(xc + w / 2))
			y2 = min(image_height - 1, int(yc + h / 2))
			boxes.append((cls_id, x1, y1, x2, y2))
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
	
	# Draw border with rounded corners effect (multiple thin lines)
	cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
	
	# Add subtle inner border for depth
	inner_color = tuple(min(255, c + 30) for c in color)
	cv2.rectangle(image, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), inner_color, 1)


def put_professional_label(image: np.ndarray, text: str, org: Tuple[int, int], 
                          color: Tuple[int, int, int], bg_color: Tuple[int, int, int] = None):
	"""Put text with professional styling including background and shadow."""
	font = cv2.FONT_HERSHEY_DUPLEX  # More professional font
	font_scale = 0.7
	thickness = 2
	
	# Get text size for background
	(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
	
	x, y = org
	
	# Ensure text stays within image bounds
	if x + text_width > image.shape[1]:
		x = image.shape[1] - text_width - 5
	if y - text_height < 0:
		y = text_height + 5
	
	# Draw background rectangle
	if bg_color is None:
		bg_color = (0, 0, 0)  # Black background by default
	
	padding = 4
	cv2.rectangle(image, 
				 (x - padding, y - text_height - padding), 
				 (x + text_width + padding, y + baseline + padding), 
				 bg_color, -1)
	
	# Draw subtle border around background
	cv2.rectangle(image, 
				 (x - padding, y - text_height - padding), 
				 (x + text_width + padding, y + baseline + padding), 
				 color, 1)
	
	# Draw text shadow for depth
	cv2.putText(image, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
	
	# Draw main text
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


def precision_recall_ap(all_gt: List[List[Tuple[int, int, int, int]]], all_preds: List[List[Tuple[Tuple[int, int, int, int], float]]], iou_threshold: float) -> Tuple[float, float, float, float]:
	"""Compute dataset-level precision, recall, F1 (at current score filtering), and AP at a given IoU.

	- all_gt: list over images of list of GT boxes (xyxy)
	- all_preds: list over images of list of ((xyxy), score) for all predictions (no additional filtering)

	Returns (precision, recall, f1, ap)
	"""
	# Build a flat list of predictions with image index for AP calculation
	preds_flat: List[Tuple[int, float, Tuple[int, int, int, int]]] = []  # (img_idx, score, box)
	gt_counts = 0
	for i, gt_boxes in enumerate(all_gt):
		gt_counts += len(gt_boxes)
		for box_score in all_preds[i]:
			box, score = box_score
			preds_flat.append((i, score, box))

	# Sort predictions by descending score
	preds_flat.sort(key=lambda x: x[1], reverse=True)

	# Track matches per image to avoid double-matching
	gt_matched: List[Dict[int, bool]] = []
	for gt_boxes in all_gt:
		gt_matched.append({j: False for j in range(len(gt_boxes))})

	# Compute TP/FP arrays for PR curve
	tp_list: List[int] = []
	fp_list: List[int] = []
	for img_idx, score, pbox in preds_flat:
		best_iou = 0.0
		best_j = -1
		for j, gbox in enumerate(all_gt[img_idx]):
			if gt_matched[img_idx][j]:
				continue
			iou = iou_xyxy(pbox, gbox)
			if iou > best_iou:
				best_iou = iou
				best_j = j
		if best_iou >= iou_threshold and best_j >= 0:
			gt_matched[img_idx][best_j] = True
			tp_list.append(1)
			fp_list.append(0)
		else:
			tp_list.append(0)
			fp_list.append(1)

	# Cumulate
	tp_cum = np.cumsum(tp_list)
	fp_cum = np.cumsum(fp_list)
	precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1)
	recall_curve = tp_cum / max(gt_counts, 1)

	# AP via precision envelope integration
	mrec = np.concatenate(([0.0], recall_curve, [1.0]))
	mpre = np.concatenate(([1.0], precision_curve, [0.0]))
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = max(mpre[i - 1], mpre[i])
	ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))

	# Point metrics at the last thresholded state
	precision = float(precision_curve[-1]) if precision_curve.size else 0.0
	recall = float(recall_curve[-1]) if recall_curve.size else 0.0
	f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
	return precision, recall, f1, ap


def create_excel_analysis(metrics_data: Dict, output_path: Path):
	"""Create Excel file with charts and analysis of detection metrics."""
	excel_path = output_path.with_suffix('.xlsx')
	
	# Create DataFrame with metrics
	df_metrics = pd.DataFrame([{
		'Metric': 'True Positives',
		'Value': metrics_data['total_tp'],
		'Description': 'Correctly detected objects'
	}, {
		'Metric': 'False Positives', 
		'Value': metrics_data['total_fp'],
		'Description': 'Incorrectly detected objects'
	}, {
		'Metric': 'False Negatives',
		'Value': metrics_data['total_fn'], 
		'Description': 'Missed objects'
	}, {
		'Metric': 'Precision',
		'Value': metrics_data['precision'],
		'Description': 'TP / (TP + FP)'
	}, {
		'Metric': 'Recall',
		'Value': metrics_data['recall'],
		'Description': 'TP / (TP + FN)'
	}, {
		'Metric': 'F1 Score',
		'Value': metrics_data['f1'],
		'Description': '2 * (Precision * Recall) / (Precision + Recall)'
	}, {
		'Metric': 'mAP@0.25',
		'Value': metrics_data['ap25'],
		'Description': 'Mean Average Precision at IoU 0.25'
	}, {
		'Metric': 'mAP@0.5',
		'Value': metrics_data['ap50'],
		'Description': 'Mean Average Precision at IoU 0.5'
	}, {
		'Metric': 'mAP@0.7',
		'Value': metrics_data['ap70'],
		'Description': 'Mean Average Precision at IoU 0.7'
	}])
	
	# Create confusion matrix data
	df_confusion = pd.DataFrame({
		'Predicted': ['Positive', 'Negative'],
		'Actual Positive': [metrics_data['total_tp'], metrics_data['total_fn']],
		'Actual Negative': [metrics_data['total_fp'], 'TN (N/A)']
	})
	
	# Create mAP comparison data
	df_map = pd.DataFrame({
		'IoU Threshold': ['0.25', '0.5', '0.7'],
		'mAP': [metrics_data['ap25'], metrics_data['ap50'], metrics_data['ap70']]
	})
	
	# Write to Excel with multiple sheets
	with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
		df_metrics.to_excel(writer, sheet_name='Metrics Summary', index=False)
		df_confusion.to_excel(writer, sheet_name='Confusion Matrix', index=False)
		df_map.to_excel(writer, sheet_name='mAP Analysis', index=False)
		
		# Create summary sheet with key statistics
		summary_data = pd.DataFrame([{
			'Dataset': metrics_data['dataset_name'],
			'Total Images': metrics_data['total_images'],
			'Total Detections': metrics_data['total_detections'],
			'Avg Inference Time (s)': metrics_data['avg_inference_time'],
			'Precision': f"{metrics_data['precision']:.4f}",
			'Recall': f"{metrics_data['recall']:.4f}",
			'F1 Score': f"{metrics_data['f1']:.4f}",
			'mAP@0.5': f"{metrics_data['ap50']:.4f}"
		}])
		summary_data.to_excel(writer, sheet_name='Executive Summary', index=False)
	
	# Create visualization charts
	plt.style.use('seaborn-v0_8')
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
	fig.suptitle(f'Detection Analysis: {metrics_data["dataset_name"]}', fontsize=16, fontweight='bold')
	
	# 1. TP/FP/FN Bar Chart
	categories = ['True Positives', 'False Positives', 'False Negatives']
	values = [metrics_data['total_tp'], metrics_data['total_fp'], metrics_data['total_fn']]
	colors = ['#2ecc71', '#e74c3c', '#f39c12']
	
	bars1 = ax1.bar(categories, values, color=colors, alpha=0.8)
	ax1.set_title('Detection Results Breakdown', fontweight='bold')
	ax1.set_ylabel('Count')
	for bar, value in zip(bars1, values):
		height = bar.get_height()
		ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
				f'{int(value)}', ha='center', va='bottom', fontweight='bold')
	
	# 2. Precision, Recall, F1 Score
	metrics_names = ['Precision', 'Recall', 'F1 Score']
	metrics_values = [metrics_data['precision'], metrics_data['recall'], metrics_data['f1']]
	colors2 = ['#3498db', '#9b59b6', '#1abc9c']
	
	bars2 = ax2.bar(metrics_names, metrics_values, color=colors2, alpha=0.8)
	ax2.set_title('Performance Metrics', fontweight='bold')
	ax2.set_ylabel('Score')
	ax2.set_ylim(0, 1)
	for bar, value in zip(bars2, metrics_values):
		height = bar.get_height()
		ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
				f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
	
	# 3. mAP at different IoU thresholds
	iou_thresholds = ['mAP@0.25', 'mAP@0.5', 'mAP@0.7']
	map_values = [metrics_data['ap25'], metrics_data['ap50'], metrics_data['ap70']]
	colors3 = ['#e67e22', '#e74c3c', '#c0392b']
	
	bars3 = ax3.bar(iou_thresholds, map_values, color=colors3, alpha=0.8)
	ax3.set_title('Mean Average Precision (mAP)', fontweight='bold')
	ax3.set_ylabel('mAP Score')
	ax3.set_ylim(0, 1)
	for bar, value in zip(bars3, map_values):
		height = bar.get_height()
		ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
				f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
	
	# 4. Pie chart for detection distribution
	detection_labels = ['True Positives', 'False Positives', 'False Negatives']
	detection_values = [metrics_data['total_tp'], metrics_data['total_fp'], metrics_data['total_fn']]
	colors4 = ['#2ecc71', '#e74c3c', '#f39c12']
	
	wedges, texts, autotexts = ax4.pie(detection_values, labels=detection_labels, colors=colors4,
									  autopct='%1.1f%%', startangle=90)
	ax4.set_title('Detection Distribution', fontweight='bold')
	
	# Make percentage text bold
	for autotext in autotexts:
		autotext.set_color('white')
		autotext.set_fontweight('bold')
	
	plt.tight_layout()
	
	# Save the chart
	chart_path = output_path.with_suffix('_analysis_charts.png')
	plt.savefig(chart_path, dpi=300, bbox_inches='tight')
	plt.close()
	
	return excel_path, chart_path


def create_per_frame_excel_analysis(frame_data: List[Dict], output_path: Path):
	"""Create Excel file with per-frame detection analysis and charts."""
	if not frame_data:
		return None, None
		
	excel_path = output_path.with_suffix('_per_frame_analysis.xlsx')
	
	# Create DataFrame from frame data
	df_frames = pd.DataFrame(frame_data)
	
	# Create summary statistics
	total_frames = len(df_frames)
	successful_detections = df_frames['Successful_Detection'].sum()
	frames_with_gt = df_frames['Has_Ground_Truth'].sum()
	frames_with_pred = df_frames['Has_Predictions'].sum()
	avg_precision = df_frames['Frame_Precision'].mean()
	avg_recall = df_frames['Frame_Recall'].mean()
	avg_inference_time = df_frames['Inference_Time_ms'].mean()
	
	summary_data = pd.DataFrame([{
		'Total_Frames': total_frames,
		'Frames_With_Ground_Truth': frames_with_gt,
		'Frames_With_Predictions': frames_with_pred,
		'Successful_Detections': successful_detections,
		'Success_Rate': successful_detections / total_frames if total_frames > 0 else 0,
		'Average_Frame_Precision': avg_precision,
		'Average_Frame_Recall': avg_recall,
		'Average_Inference_Time_ms': avg_inference_time
	}])
	
	# Write to Excel with multiple sheets
	with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
		df_frames.to_excel(writer, sheet_name='Frame_by_Frame_Analysis', index=False)
		summary_data.to_excel(writer, sheet_name='Summary_Statistics', index=False)
		
		# Create detection success timeline
		detection_timeline = df_frames[['Frame_Number', 'Successful_Detection', 'Ground_Truth_Count', 'Prediction_Count']].copy()
		detection_timeline.to_excel(writer, sheet_name='Detection_Timeline', index=False)
	
	# Create visualization charts
	plt.style.use('seaborn-v0_8')
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
	fig.suptitle('Per-Frame Detection Analysis', fontsize=16, fontweight='bold')
	
	# 1. Detection Success Timeline
	ax1.plot(df_frames['Frame_Number'], df_frames['Successful_Detection'].astype(int), 
			 'o-', color='#2ecc71', markersize=3, linewidth=1.5, alpha=0.7)
	ax1.fill_between(df_frames['Frame_Number'], df_frames['Successful_Detection'].astype(int), 
					alpha=0.3, color='#2ecc71')
	ax1.set_title('Detection Success Timeline', fontweight='bold')
	ax1.set_xlabel('Frame Number')
	ax1.set_ylabel('Successful Detection (1=Yes, 0=No)')
	ax1.grid(True, alpha=0.3)
	ax1.set_ylim(-0.1, 1.1)
	
	# 2. Ground Truth vs Predictions Count
	ax2.scatter(df_frames['Ground_Truth_Count'], df_frames['Prediction_Count'], 
			   c=df_frames['Successful_Detection'], cmap='RdYlGn', alpha=0.6, s=50)
	ax2.set_title('Ground Truth vs Predictions', fontweight='bold')
	ax2.set_xlabel('Ground Truth Count')
	ax2.set_ylabel('Prediction Count')
	ax2.grid(True, alpha=0.3)
	
	# Add diagonal line for perfect detection
	max_count = max(df_frames['Ground_Truth_Count'].max(), df_frames['Prediction_Count'].max())
	ax2.plot([0, max_count], [0, max_count], '--', color='gray', alpha=0.5, label='Perfect Detection')
	ax2.legend()
	
	# 3. Detection Success Distribution
	success_counts = df_frames['Successful_Detection'].value_counts()
	labels = ['Failed Detection', 'Successful Detection']
	colors = ['#e74c3c', '#2ecc71']
	sizes = [success_counts.get(False, 0), success_counts.get(True, 0)]
	
	wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
									  startangle=90)
	ax3.set_title('Detection Success Distribution', fontweight='bold')
	
	# Make percentage text bold and white
	for autotext in autotexts:
		autotext.set_color('white')
		autotext.set_fontweight('bold')
	
	# 4. Frame-level Precision and Recall
	frames_with_metrics = df_frames[(df_frames['Frame_Precision'] > 0) | (df_frames['Frame_Recall'] > 0)]
	if not frames_with_metrics.empty:
		ax4.scatter(frames_with_metrics['Frame_Precision'], frames_with_metrics['Frame_Recall'], 
				   alpha=0.6, s=50, color='#3498db')
		ax4.set_title('Frame-level Precision vs Recall', fontweight='bold')
		ax4.set_xlabel('Frame Precision')
		ax4.set_ylabel('Frame Recall')
		ax4.grid(True, alpha=0.3)
		ax4.set_xlim(0, 1)
		ax4.set_ylim(0, 1)
		
		# Add diagonal line for F1 contours
		x = np.linspace(0, 1, 100)
		for f1 in [0.2, 0.4, 0.6, 0.8]:
			y = f1 * x / (2 * x - f1)
			y = np.where((y >= 0) & (y <= 1), y, np.nan)
			ax4.plot(x, y, '--', alpha=0.3, color='gray')
	else:
		ax4.text(0.5, 0.5, 'No frames with\nprecision/recall data', 
				ha='center', va='center', transform=ax4.transAxes, fontsize=12)
		ax4.set_title('Frame-level Precision vs Recall', fontweight='bold')
	
	plt.tight_layout()
	
	# Save the chart
	chart_path = output_path.with_suffix('_per_frame_charts.png')
	plt.savefig(chart_path, dpi=300, bbox_inches='tight')
	plt.close()
	
	return excel_path, chart_path


def run(
	images_dir: Path,
	labels_dir: Path,
	model_weights_dir: Path,
	output_video_path: Path,
	confidence_threshold: float = 0.25,
	image_size: int = 640,
	skip_invalid_labels: bool = True,
	max_box_area_ratio: float = 0.8,
	save_per_frame: bool = False,
):
	# Resolve model weights: prefer best.pt else last.pt
	best = model_weights_dir / "best.pt"
	last = model_weights_dir / "last.pt"
	weights_path = best if best.exists() else last
	if not weights_path.exists():
		raise FileNotFoundError(f"No weights found at {best} or {last}")

	model = YOLO(str(weights_path))
	model.conf = confidence_threshold

	# Collect images (jpg/png, sorted)
	image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
	if not image_files:
		raise FileNotFoundError(f"No images found in {images_dir}")

	# Prepare video writer using first image size
	first_img = cv2.imread(str(image_files[0]))
	if first_img is None:
		raise RuntimeError(f"Failed to read first image: {image_files[0]}")
	height, width = first_img.shape[:2]
	fps = 3.0
	output_video_path.parent.mkdir(parents=True, exist_ok=True)
	writer = cv2.VideoWriter(
		str(output_video_path),
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps,
		(width, height),
	)

	# Professional color scheme (BGR format)
	red = (0, 0, 255)      # Red for ground truth boxes
	green = (0, 255, 0)    # Bright green for predictions
	text_color = (255, 255, 255)  # White text
	bg_color = (0, 0, 0)   # Black background for text

	# Create frames subfolder if per-frame saving is enabled
	frames_dir = None
	frame_detection_data = []
	if save_per_frame:
		frames_dir = output_video_path.parent / "frames"
		frames_dir.mkdir(parents=True, exist_ok=True)
		print(f"📁 Created frames directory: {frames_dir}")

	# Metrics accumulators
	all_gt_boxes: List[List[Tuple[int, int, int, int]]] = []
	all_pred_boxes_all_scores: List[List[Tuple[Tuple[int, int, int, int], float]]] = []
	# For point metrics at conf threshold (TP/FP/FN)
	total_tp = 0
	total_fp = 0
	total_fn = 0
	infer_times: List[float] = []
	skipped_frames = 0

	for img_path in image_files:
		frame = cv2.imread(str(img_path))
		if frame is None:
			continue
		ih, iw = frame.shape[:2]

		# Ground truth from label file
		lbl_path = labels_dir / (img_path.stem + ".txt")
		gt_boxes = load_yolo_txt_labels(lbl_path, iw, ih)
		
		# Check for invalid/suspicious labels (boxes covering >80% of image)
		if skip_invalid_labels:
			invalid = False
			for cls_id, x1, y1, x2, y2 in gt_boxes:
				box_area = (x2 - x1) * (y2 - y1)
				image_area = iw * ih
				area_ratio = box_area / image_area if image_area > 0 else 0
				if area_ratio > max_box_area_ratio:
					print(f"[SKIP] {img_path.name}: GT box covers {area_ratio*100:.1f}% of image (threshold: {max_box_area_ratio*100:.1f}%)")
					invalid = True
					break
			if invalid:
				skipped_frames += 1
				continue
		
		all_gt_boxes.append([(x1, y1, x2, y2) for _, x1, y1, x2, y2 in gt_boxes])

		# Prediction
		start = time.perf_counter()
		results = model.predict(source=frame, imgsz=image_size, verbose=False)[0]
		infer_times.append(time.perf_counter() - start)
		pred_boxes = []
		if len(results.boxes) > 0:
			xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
			scores = results.boxes.conf.cpu().numpy()
			for (x1, y1, x2, y2), s in zip(xyxy, scores):
				pred_boxes.append(((x1, y1, x2, y2), float(s)))
		all_pred_boxes_all_scores.append(pred_boxes)

		# Draw GT boxes: red outline only (no fill)
		for cls_id, x1, y1, x2, y2 in gt_boxes:
			box = (x1, y1, x2, y2)
			draw_professional_box(frame, box, red, alpha=0.3, thickness=3, fill=False)
			put_professional_label(frame, "drone (gt)", (x1, max(25, y1 - 10)), text_color, bg_color)

		# Draw predicted boxes: green with professional styling
		for (x1, y1, x2, y2), score in pred_boxes:
			draw_professional_box(frame, (x1, y1, x2, y2), green, alpha=0.2, thickness=3)
			put_professional_label(frame, f"drone: {score:.2f}", (x1, max(25, y1 - 10)), text_color, bg_color)

		# Count TP/FP/FN at IoU 0.5 using predictions above current model.conf
		pred_filtered = [((x1, y1, x2, y2), s) for ((x1, y1, x2, y2), s) in pred_boxes if s >= confidence_threshold]
		matched_gt = [False] * len(all_gt_boxes[-1])
		for (px1, py1, px2, py2), s in sorted(pred_filtered, key=lambda t: t[1], reverse=True):
			best_iou = 0.0
			best_idx = -1
			for j, gbox in enumerate(all_gt_boxes[-1]):
				if matched_gt[j]:
					continue
				iou = iou_xyxy((px1, py1, px2, py2), gbox)
				if iou > best_iou:
					best_iou = iou
					best_idx = j
			if best_iou >= 0.5 and best_idx >= 0:
				matched_gt[best_idx] = True
				total_tp += 1
			else:
				total_fp += 1
		# Remaining unmatched GTs are FN
		for used in matched_gt:
			if not used:
				total_fn += 1

		# Add professional frame info display
		frame_num = image_files.index(img_path) + 1
		total_frames = len(image_files)
		gt_count = len(gt_boxes)
		pred_count = len([p for p in pred_boxes if p[1] >= confidence_threshold])
		
		# Frame info in top-left corner
		frame_info = f"Frame: {frame_num}/{total_frames} | GT: {gt_count} | Pred: {pred_count}"
		put_professional_label(frame, frame_info, (10, 30), text_color, bg_color)
		
		# Image name in top-right corner
		text_width = cv2.getTextSize(img_path.name, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0][0]
		put_professional_label(frame, img_path.name, (iw - text_width - 20, 30), text_color, bg_color)
		
		# Inference time in bottom-left corner
		if infer_times:
			current_infer_time = infer_times[-1]
			time_info = f"Inference: {current_infer_time*1000:.1f}ms"
			put_professional_label(frame, time_info, (10, ih - 20), text_color, bg_color)

		# Save individual frame if per-frame saving is enabled
		if save_per_frame and frames_dir:
			frame_filename = f"frame_{frame_num:06d}_{img_path.stem}.jpg"
			frame_path = frames_dir / frame_filename
			cv2.imwrite(str(frame_path), frame)
			
			# Track detection success for this frame
			has_gt = len(gt_boxes) > 0
			has_pred = pred_count > 0
			has_successful_detection = has_gt and has_pred and total_tp > 0  # At least one TP in current state
			
			# Calculate frame-level TP/FP/FN
			frame_tp = sum(1 for used in matched_gt if used)
			frame_fp = pred_count - frame_tp
			frame_fn = gt_count - frame_tp
			
			frame_detection_data.append({
				'Frame_Number': frame_num,
				'Image_Name': img_path.name,
				'Frame_File': frame_filename,
				'Ground_Truth_Count': gt_count,
				'Prediction_Count': pred_count,
				'True_Positives': frame_tp,
				'False_Positives': frame_fp,
				'False_Negatives': frame_fn,
				'Has_Ground_Truth': has_gt,
				'Has_Predictions': has_pred,
				'Successful_Detection': has_successful_detection,
				'Inference_Time_ms': current_infer_time * 1000 if infer_times else 0,
				'Frame_Precision': frame_tp / (frame_tp + frame_fp) if (frame_tp + frame_fp) > 0 else 0,
				'Frame_Recall': frame_tp / (frame_tp + frame_fn) if (frame_tp + frame_fn) > 0 else 0
			})

		# Ensure frame size consistent
		if (frame.shape[1], frame.shape[0]) != (width, height):
			frame = cv2.resize(frame, (width, height))

		writer.write(frame)

	writer.release()

	# Dataset-level metrics
	total_images = len(image_files)
	total_detections = int(sum(len(p) for p in all_pred_boxes_all_scores))
	precision_point = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
	recall_point = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
	f1_point = (2 * precision_point * recall_point / (precision_point + recall_point)) if (precision_point + recall_point) > 0 else 0.0
	avg_infer = float(np.mean(infer_times)) if infer_times else 0.0

	# AP at custom IoUs
	_, _, _, ap25 = precision_recall_ap(all_gt_boxes, all_pred_boxes_all_scores, 0.25)
	_, _, _, ap50 = precision_recall_ap(all_gt_boxes, all_pred_boxes_all_scores, 0.50)
	_, _, _, ap70 = precision_recall_ap(all_gt_boxes, all_pred_boxes_all_scores, 0.70)

	# Prepare metrics data for Excel analysis
	metrics_data = {
		'dataset_name': images_dir.name,
		'total_images': total_images,
		'total_detections': total_detections,
		'total_tp': total_tp,
		'total_fp': total_fp,
		'total_fn': total_fn,
		'precision': precision_point,
		'recall': recall_point,
		'f1': f1_point,
		'ap25': ap25,
		'ap50': ap50,
		'ap70': ap70,
		'avg_inference_time': avg_infer
	}
	
	# Create Excel analysis with charts
	try:
		excel_path, chart_path = create_excel_analysis(metrics_data, output_video_path)
		print(f"📊 Created Excel analysis: {excel_path}")
		print(f"📈 Created analysis charts: {chart_path}")
	except Exception as e:
		print(f"⚠️  Warning: Could not create Excel analysis: {str(e)}")
		# Fallback to CSV
		csv_path = output_video_path.with_suffix("")
		csv_path = csv_path.parent / f"{csv_path.name}_metrics.csv"
		with open(csv_path, "w", newline="") as f:
			writer_csv = csv.writer(f)
			writer_csv.writerow([
				"dataset",
				"total_images",
				"total_detections",
				"total_true_positives",
				"total_false_positives",
				"total_false_negatives",
				"precision",
				"recall",
				"f1_score",
				"mAP@0.25",
				"mAP@0.5",
				"mAP@0.7",
				"avg_inference_time_s",
			])
			writer_csv.writerow([
				images_dir.name,
				total_images,
				total_detections,
				total_tp,
				total_fp,
				total_fn,
				f"{precision_point:.4f}",
				f"{recall_point:.4f}",
				f"{f1_point:.4f}",
				f"{ap25:.4f}",
				f"{ap50:.4f}",
				f"{ap70:.4f}",
				f"{avg_infer:.4f}",
			])
		print(f"📄 Saved fallback CSV: {csv_path}")

	print(f"\n🎬 Saved annotated video to: {output_video_path}")
	print("\n📊 Metrics Summary:")
	print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	print(f"📁 Dataset: {images_dir.name}")
	print(f"🖼️  Total images found: {len(image_files)} | ⚠️  Skipped (invalid): {skipped_frames} | ✅ Processed: {total_images}")
	print(f"🎯 Total detections: {total_detections}")
	print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	print(f"✅ True Positives (TP): {total_tp}")
	print(f"❌ False Positives (FP): {total_fp}")
	print(f"⭕ False Negatives (FN): {total_fn}")
	print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	print(f"🎯 Precision: {precision_point:.4f} ({precision_point*100:.2f}%)")
	print(f"🔍 Recall: {recall_point:.4f} ({recall_point*100:.2f}%)")
	print(f"⚖️  F1 Score: {f1_point:.4f} ({f1_point*100:.2f}%)")
	print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	print(f"📈 mAP@0.25: {ap25:.4f} ({ap25*100:.2f}%)")
	print(f"📈 mAP@0.50: {ap50:.4f} ({ap50*100:.2f}%)")
	print(f"📈 mAP@0.70: {ap70:.4f} ({ap70*100:.2f}%)")
	print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	print(f"⏱️  Average inference time: {avg_infer:.4f}s ({avg_infer*1000:.1f}ms)")
	print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	# Create per-frame analysis if enabled
	if save_per_frame and frame_detection_data:
		try:
			per_frame_excel, per_frame_chart = create_per_frame_excel_analysis(frame_detection_data, output_video_path)
			if per_frame_excel:
				print(f"\n🖼️  Per-Frame Analysis:")
				print(f"📊 Created per-frame Excel: {per_frame_excel}")
				print(f"📈 Created per-frame charts: {per_frame_chart}")
				
				# Print per-frame summary
				df_summary = pd.DataFrame(frame_detection_data)
				successful_frames = df_summary['Successful_Detection'].sum()
				total_saved_frames = len(df_summary)
				success_rate = successful_frames / total_saved_frames if total_saved_frames > 0 else 0
				
				print(f"🎯 Frames with successful detection: {successful_frames}/{total_saved_frames} ({success_rate*100:.1f}%)")
				print(f"📁 Individual frames saved to: {frames_dir}")
		except Exception as e:
			print(f"⚠️  Warning: Could not create per-frame analysis: {str(e)}")


def parse_args():
	parser = argparse.ArgumentParser(description="Create professional video with GT (red outline) and predictions (green) from images+labels using YOLO, with Excel analysis")
	parser.add_argument(
		"--images-dir", 
		type=Path, 
		default=Path(r"F:\Data collection - real footage\Kraken_label.v1i.yolov11\train\images"),
		help="Directory containing images"
	)
	parser.add_argument(
		"--labels-dir", 
		type=Path, 
		default=Path(r"F:\Data collection - real footage\Kraken_label.v1i.yolov11\train\labels"),
		help="Directory containing YOLO label files"
	)
	parser.add_argument(
		"--model-weights-dir", 
		type=Path, 
		default=Path(r"D:\SiDG-ATRID-Dataset\Model_test_results\warsaw_model_yolo11m_w_trainingset\warsaw_model_yolo11m\warsaw_dataset_model\weights"),
		help="Directory containing best.pt/last.pt"
	)
	parser.add_argument(
		"--output", 
		type=Path, 
		default=Path(r"F:\Data collection - real footage\Kraken_label.v1i.yolov11\sidgatrid-kraken_video_test.mp4"),
		help="Output MP4 path"
	)
	parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
	parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
	parser.add_argument("--skip-invalid-labels", action="store_true", default=True, help="Skip frames with GT boxes covering >80%% of image")
	parser.add_argument("--max-box-area-ratio", type=float, default=0.8, help="Max allowed GT box area ratio (default: 0.8)")
	parser.add_argument("--per-frame-img", action="store_true", help="Save each processed frame as image in 'frames' subfolder with detection tracking Excel")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(
		images_dir=args.images_dir,
		labels_dir=args.labels_dir,
		model_weights_dir=args.model_weights_dir,
		output_video_path=args.output,
		confidence_threshold=args.conf,
		image_size=args.imgsz,
		skip_invalid_labels=args.skip_invalid_labels,
		max_box_area_ratio=args.max_box_area_ratio,
		save_per_frame=args.per_frame_img,
	)


