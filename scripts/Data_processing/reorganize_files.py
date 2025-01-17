import os
import shutil

def reorganize_images(prepared_dataset_dir, dataset_dir, unrealcvoutput_dir):
    """
    Reorganize images from the prepared dataset back to their weather-specific folders.

    Args:
        prepared_dataset_dir (str): Path to the prepared dataset folder.
        dataset_dir (str): Path to the dataset/images/train folder where weather-specific folders exist.
        unrealcvoutput_dir (str): Path to the unrealcvoutput folder where images should be copied.
    """
    # Input folders
    prepared_train_images_dir = os.path.join(prepared_dataset_dir, "train", "images")
    prepared_val_images_dir = os.path.join(prepared_dataset_dir, "val", "images")

    # Weather-specific source folders under dataset_dir
    weather_folders = ["clearsky", "sunnyclear", "nightcloudy", "sunnycloudy"]
    dataset_weather_dirs = {folder: os.path.join(dataset_dir, "images", "train", folder) for folder in weather_folders}

    # Destination folders in unrealcvoutput_dir
    unrealcv_weather_dirs = {folder: os.path.join(unrealcvoutput_dir, folder, "rgb") for folder in weather_folders}

    # Ensure destination folders exist
    for weather_dir in unrealcv_weather_dirs.values():
        os.makedirs(weather_dir, exist_ok=True)

    # Helper function to find weather folder for an image
    def find_weather_folder(filename):
        for weather, source_dir in dataset_weather_dirs.items():
            if os.path.exists(os.path.join(source_dir, filename)):
                return weather
        return None  # If not found in any weather folder

    # Process train and val images
    all_images_dirs = [prepared_train_images_dir, prepared_val_images_dir]
    for images_dir in all_images_dirs:
        for filename in os.listdir(images_dir):
            if filename.endswith(".png"):
                # Determine weather folder
                weather = find_weather_folder(filename)
                if not weather:
                    print(f"Could not find weather folder for {filename}. Skipping.")
                    continue

                # Source image path
                source_image_path = os.path.join(images_dir, filename)

                # Destination image path
                dest_image_path = os.path.join(unrealcv_weather_dirs[weather], filename)

                # Copy image
                shutil.copy(source_image_path, dest_image_path)
                print(f"Copied {filename} to {unrealcv_weather_dirs[weather]}.")

    print("Reorganization complete!")

# Example usage
if __name__ == "__main__":
    # Paths
    prepared_dataset_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\prepared_dataset"
    dataset_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput\dataset"
    unrealcvoutput_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\data\unrealcvoutput"

    # Reorganize images
    reorganize_images(prepared_dataset_dir, dataset_dir, unrealcvoutput_dir)
