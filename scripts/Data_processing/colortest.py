import cv2
import numpy as np

def list_colors_in_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Reshape the image to a 2D array of pixels (height * width, channels)
    reshaped = image.reshape(-1, image.shape[-1])

    # Get unique colors as a set of tuples
    unique_colors = np.unique(reshaped, axis=0)

    # Convert to a list of tuples for easier readability
    color_list = [tuple(color) for color in unique_colors]

    return color_list

# Example usage
image_path = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\sunnycloudy2\mask\ .png"
colors = list_colors_in_image(image_path)

# Print all unique colors
print("Unique colors in the image:")
for color in colors:
    print(color)
