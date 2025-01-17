import os
import cv2
import json
import numpy as np

class YOLODatasetGenerator:
    def __init__(self, mask_dir, rgb_dir, json_file, output_img_dir, output_label_dir):
        self.mask_dir = mask_dir
        self.rgb_dir = rgb_dir
        self.json_file = json_file
        self.output_img_dir = output_img_dir
        self.output_label_dir = output_label_dir

        # Ensure output directories exist
        os.makedirs(self.output_img_dir, exist_ok=True)
        os.makedirs(self.output_label_dir, exist_ok=True)

        # Load object-to-color mapping
        self.color_mapping = self.load_color_mapping()

    # def load_color_mapping(self):
    #     """Load object-to-color mapping from the JSON file."""
    #     with open(self.json_file, "r") as f:
    #         color_mapping = json.load(f)
    #     # Convert to {color_tuple: (class_id, object_name)} format
    #     return {
    #         (v["R"], v["G"], v["B"]): (idx, k)  # Extract R, G, B values as tuple
    #         for idx, (k, v) in enumerate(color_mapping.items())
    #     }
    
    def load_color_mapping(self):
        return {  # Example mapping with colors and class IDs
            (127, 63, 255): (0, "DJIOctocopter"),  # Adjusted to match your detected colors
        }

    def process_masks(self, object_names):
        for mask_filename in os.listdir(self.mask_dir):
            if mask_filename.endswith("_object_mask.png"):
                # Load mask and corresponding RGB image
                mask_path = os.path.join(self.mask_dir, mask_filename)
                mask_img = cv2.imread(mask_path)

                rgb_filename = mask_filename.replace("_object_mask", "_lit")
                rgb_path = os.path.join(self.rgb_dir, rgb_filename)
                rgb_img = cv2.imread(rgb_path)

                if mask_img is None:
                    print(f"Failed to load mask image: {mask_path}")
                    continue
                if rgb_img is None:
                    print(f"Failed to load RGB image: {rgb_path}")
                    continue

                # Initialize YOLO annotation list
                annotations = []

                for color, (class_id, obj_name) in self.color_mapping.items():
                    if obj_name not in object_names:
                        continue  # Skip objects not in the target list

                    # Use a tolerance range for color matching
                    tolerance = 20
                    lower_bound = np.array([max(0, c - tolerance) for c in color], dtype=np.uint8)
                    upper_bound = np.array([min(255, c + tolerance) for c in color], dtype=np.uint8)

                    # Create binary mask for the current color
                    binary_mask = cv2.inRange(mask_img, lower_bound, upper_bound)

                    # Find all non-zero pixels
                    points = cv2.findNonZero(binary_mask)
                    if points is not None:
                        x, y, w, h = cv2.boundingRect(points)
                        if w == 0 or h == 0:
                            print(f"Invalid bounding box for {obj_name} in {mask_filename}")
                            continue

                        # Normalize coordinates for YOLO format
                        img_h, img_w = mask_img.shape[:2]
                        x_center = (x + w / 2) / img_w
                        y_center = (y + h / 2) / img_h
                        norm_width = w / img_w
                        norm_height = h / img_h

                        # Add to annotations
                        annotations.append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")

                        # Draw bounding box and label on the RGB image
                        cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(rgb_img, obj_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        print(f"No pixels found for {obj_name} in {mask_filename}")

                # Save YOLO annotations
                if annotations:
                    label_filename = mask_filename.replace("_object_mask.png", ".txt")
                    label_path = os.path.join(self.output_label_dir, label_filename)
                    with open(label_path, "w") as label_file:
                        label_file.write("\n".join(annotations))

                    # Save labeled RGB image
                    output_rgb_path = os.path.join(self.output_img_dir, rgb_filename)
                    cv2.imwrite(output_rgb_path, rgb_img)

        print(f"Processing complete! Annotations saved to {self.output_label_dir} and labeled images to {self.output_img_dir}.")

# Usage Example
if __name__ == "__main__":
    # Parameters
    mask_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\clearsky\mask"
    rgb_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\clearsky\rgb"
    json_file = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\clearsky\agent_color_info.json"
    output_img_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\dataset\images\train\clearsky"
    output_label_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\dataset\labels\train\clearsky"

    # Target objects of interest
    target_objects = ["DJIOctocopter"]

    # Initialize and run the generator
    generator = YOLODatasetGenerator(mask_dir, rgb_dir, json_file, output_img_dir, output_label_dir)
    generator.process_masks(target_objects)
