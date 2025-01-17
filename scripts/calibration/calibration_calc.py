import numpy as np
import cv2 as cv
import os
import math

# Define the dimension of the calibration patterns
# gridSize = (19, 14)  # (columns, rows)
gridSize = (9, 9)  # (columns, rows)

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Directory paths
root_dir = r"C:\Users\Josh\Desktop\Masters\Research\SiDG-ATRID\git\SIDG-ATRID\data"
int_dir = root_dir + r"\simulation_camera\ImageStream"
output_dir = root_dir + r"\output"

# Check directories
if not os.path.exists(int_dir):
    print(f"Make sure the data folder {int_dir} exists")
else:
    print(f"Folder already exists at: {int_dir}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder created at: {output_dir}")
else:
    print(f"Folder already exists at: {output_dir}")

# Blob detector parameters
circleColor = 1  # 1 for white circle and 0 for black circle

# Setup SimpleBlobDetector parameters
blobParams = cv.SimpleBlobDetector_Params()
blobParams.minThreshold = 20.0
blobParams.maxThreshold = 240.0
blobParams.thresholdStep = 5.0
blobParams.minDistBetweenBlobs = 5.0

blobParams.filterByArea = True
blobParams.minArea = 20.0
blobParams.maxArea = 90000.0

blobParams.filterByCircularity = True
blobParams.minCircularity = 0.4
blobParams.maxCircularity = 1.0

blobParams.filterByConvexity = True
blobParams.minConvexity = 0.4
blobParams.maxConvexity = 1.0

blobParams.filterByColor = True
blobParams.blobColor = 255 * circleColor

# Create a blob detector with the parameters
blobDetector = cv.SimpleBlobDetector_create(blobParams)

# Prepare object points
objp = np.zeros((gridSize[1]*gridSize[0], 3), np.float32)
circleCenterDist = 25.4/264 * 125  # Circle spacing, adjust as needed
objp[:, :2] = np.mgrid[0:gridSize[0], 0:gridSize[1]].T.reshape(-1, 2) * circleCenterDist

# Create arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Get images from directory
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
images = [file for file in os.listdir(int_dir) if file.endswith(image_extensions)]
num_images = len(images)
print(f"The amount of images for intrinsic calibration: {num_images}")

for i, image_file in enumerate(images):
    print(f"Processing image {i + 1} / {num_images}")
    image_path = os.path.join(int_dir, image_file)
    RawImg = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    if RawImg is None:
        print("Error: Image not found")
        continue

    # Detect the circle grid
    found, tempCenters = cv.findChessboardCorners(RawImg, gridSize, None, cv.CALIB_CB_CLUSTERING | cv.CALIB_CB_SYMMETRIC_GRID, blobDetector)

    if found:
        objpoints.append(objp)
        imgpoints.append(tempCenters)
        
        # Convert to color image for visualization
        visImg = cv.cvtColor(RawImg, cv.COLOR_GRAY2BGR)

        # Draw the circle centers on the image for debugging
        for center in tempCenters:
            # Convert the center coordinates to integers
            center_int = (int(center[0][0]), int(center[0][1]))
            cv.circle(visImg, center_int, 5, (0, 255, 0), 2)  # Draw green circles on detected centers

        # Draw the complete grid for debugging
        cv.drawChessboardCorners(visImg, gridSize, tempCenters, found)

        # Display and save the visualized image with debug lines
        output_image_path = os.path.join(output_dir, f"debug_visualized_{i}.png")
        cv.imwrite(output_image_path, visImg)
        print(f"Saved visualization for image {i + 1}")

# Calibrate the camera if enough points were found
if len(objpoints) > 0 and len(imgpoints) > 0:
    print("Performing camera calibration...")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, RawImg.shape[::-1], None, None)
    
    # Reprojection error calculation
    tot_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)
        tot_error += error
        total_points += len(objpoints[i])
    
    meanError = math.sqrt(tot_error / total_points)
    print(f"Mean reprojection error: {meanError}")

    # Save camera matrix and distortion coefficients
    IntMatrixFile = os.path.join(output_dir, "CamIntrinsicMatrix.txt")
    DistVecFile = os.path.join(output_dir, "CamDistortionCoeff.txt")
    
    np.savetxt(IntMatrixFile, mtx, fmt='%.6f', delimiter=',')
    np.savetxt(DistVecFile, dist, fmt='%.6f', delimiter=',')
    
    print(f"Camera matrix saved to {IntMatrixFile}")
    print(f"Distortion coefficients saved to {DistVecFile}")
else:
    print("Not enough points found for camera calibration.")
