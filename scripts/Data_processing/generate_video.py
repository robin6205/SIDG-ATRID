import os
import cv2

def extract_index(filename):
    """
    Extract the index from the filename. Assumes the format 'index_date_time_something.png'.

    Args:
        filename (str): The filename of the image.

    Returns:
        int: The extracted index as an integer.
    """
    try:
        # Split the filename by underscores and take the first part as the index
        return int(filename.split("_")[0])
    except ValueError:
        return float('inf')  # Return a large number to push invalid files to the end

def create_video_from_images(image_dir, output_video_path, frame_rate=10):
    """
    Create a video from images in a directory, sorted by index.

    Args:
        image_dir (str): Directory containing images.
        output_video_path (str): Path to save the output video.
        frame_rate (int): Frame rate for the video (frames per second).
    """
    # Get a list of PNG files in the directory
    images = [img for img in os.listdir(image_dir) if img.endswith(".png")]

    # Sort the images based on the extracted index
    images.sort(key=extract_index)

    # Prepend the directory path to each file
    images = [os.path.join(image_dir, img) for img in images]

    # Check if there are images to process
    if not images:
        print(f"No PNG images found in directory: {image_dir}")
        return

    # Read the first image to get the frame size
    first_image = cv2.imread(images[0])
    height, width, _ = first_image.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4v codec
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    print(f"Creating video at {output_video_path}...")
    for img_path in images:
        # Read each image
        img = cv2.imread(img_path)

        # Check if the image is the same size as the first frame
        if img.shape[:2] != (height, width):
            print(f"Skipping {img_path}: size mismatch with the first frame.")
            continue

        # Write the frame to the video
        video_writer.write(img)

    # Release the video writer
    video_writer.release()
    print(f"Video creation complete: {output_video_path}")

# Usage example
if __name__ == "__main__":
    # Define paths and parameters
    image_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\dataset\images\train\sunnycloudy"
    output_video_path = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\scripts\output_video_sunnycloudy.mp4"
    frame_rate = 30  # Frames per second

    # Create the video
    create_video_from_images(image_dir, output_video_path, frame_rate)
