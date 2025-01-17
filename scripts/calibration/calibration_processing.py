import cv2
import os

# Parameters
video_path = r"C:\Users\Josh\Desktop\Masters\Research\SiDG-ATRID\git\SIDG-ATRID\data\Bosche_ipfusion9000i_cam\Calibration_Test02.mp4"
output_dir = r"C:\Users\Josh\Desktop\Masters\Research\SiDG-ATRID\git\SIDG-ATRID\data\Bosche_ipfusion9000i_cam\photos"
frame_rate = 2  # Sample every 2 frames
left_crop = 10  # Crop 10 pixels from the left
right_crop = 500  # Crop 500 pixels from the right
threshold_value = 150  # Threshold value for binarization

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Video capture
cap = cv2.VideoCapture(video_path)

frame_index = 0
photo_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    if frame_index % frame_rate == 0:
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Crop the image
        cropped_frame = gray_frame[:, left_crop:-right_crop]

        # Apply thresholding to make the background black and circles white
        _, thresholded_frame = cv2.threshold(cropped_frame, threshold_value, 255, cv2.THRESH_BINARY)

        # Save the image
        output_path = os.path.join(output_dir, f"photo_{photo_index}.png")
        cv2.imwrite(output_path, thresholded_frame)
        
        # Display the first frame to adjust parameters if needed
        if photo_index == 0:
            cv2.imshow('First Processed Frame', thresholded_frame)
            cv2.waitKey(0)  # Wait for key press to continue
        
        photo_index += 1
    
    frame_index += 1

cap.release()
cv2.destroyAllWindows()
