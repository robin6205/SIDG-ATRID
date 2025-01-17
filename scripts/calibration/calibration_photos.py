import cv2
import os

# Parameters
video_path = r"C:\Users\Josh\Desktop\Masters\Research\SiDG-ATRID\git\SIDG-ATRID\data\Bosche_ipfusion9000i_cam\Calibration_Test02.mp4"
output_dir = r"C:\Users\Josh\Desktop\Masters\Research\SiDG-ATRID\git\SIDG-ATRID\data\Bosche_ipfusion9000i_cam\photos"
frame_rate = 2  # Save every 2 frames

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory at: {output_dir}")
else:
    print(f"Directory already exists at: {output_dir}")

# Video capture
cap = cv2.VideoCapture(video_path)

frame_index = 0
photo_index = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    # Save every `frame_rate` frames
    if frame_index % frame_rate == 0:
        # Save the frame as an image
        output_path = os.path.join(output_dir, f"photo_{photo_index}.png")
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")
        
        # Optionally display the frame
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)  # Short pause to display the frame
        
        photo_index += 1
    
    frame_index += 1

# Release video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
