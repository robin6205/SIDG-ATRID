import os
import cv2
import torch
import matplotlib.pyplot as plt

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the directory for testing images
rgb_image_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\clearsky\rgb"

# Set the output directory
output_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\clearsky\yolo"
os.makedirs(output_dir, exist_ok=True)

# Iterate over images in the directory
index = 0
for image_file in os.listdir(rgb_image_dir):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(rgb_image_dir, image_file)

        # Load and process image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model(img)

        # Print results
        results.print()
        
        # Render and save results
        results.render()  # Render results on the image
        rendered_img = results.ims[0]  # Get the rendered image
        output_path = os.path.join(output_dir, f'yolo_{index}.jpg')
        rendered_img_bgr = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
        cv2.imwrite(output_path, rendered_img_bgr)
        print(f'Saved output image to {output_path}')
        
        # Optional: Display single plot with matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(rendered_img)
        plt.axis('off')
        plt.title(f'Detection Results for {image_file}')
        plt.show()
        
        index += 1