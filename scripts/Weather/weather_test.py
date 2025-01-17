import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models
import torch.nn as nn

class MultiTaskResNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskResNet, self).__init__()
        self.backbone = models.resnet50(weights=None)  # No need for pretrained weights during inference
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        classification = self.classifier(features)
        regression = self.regressor(features)
        return classification, regression

def predict_weather(image_path, model_path):
    """
    Predict weather class and temperature for a given image.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the trained model (.pth file)
    
    Returns:
        tuple: (predicted_class, predicted_temperature)
    """
    # Define class labels
    classes = ['cloudy', 'foggy', 'rain', 'snow', 'sunny']
    num_classes = len(classes)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare the model
    model = MultiTaskResNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    # Perform inference
    with torch.no_grad():
        try:
            class_output, temp_output = model(image_tensor)
            
            # Get predicted class
            _, predicted_class = torch.max(class_output, 1)
            predicted_class_idx = predicted_class.item()
            predicted_class_name = classes[predicted_class_idx]
            
            # Get class probabilities
            class_probabilities = torch.nn.functional.softmax(class_output[0], dim=0)
            
            # Get predicted temperature
            predicted_temp = temp_output.item()
            
            # Print detailed results
            print("\nPrediction Results:")
            print("-" * 20)
            print(f"Predicted Weather: {predicted_class_name}")
            print("\nClass Probabilities:")
            for i, class_name in enumerate(classes):
                probability = class_probabilities[i].item() * 100
                print(f"{class_name}: {probability:.2f}%")
            print(f"\nPredicted Temperature: {predicted_temp:.1f}°C")
            
            return predicted_class_name, predicted_temp
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return None, None

def main():
    # Define paths
    model_path = "weather_classifier_regressor_resnet50.pth"  # Update with your model path
    image_path = "C:/Users/Josh/Desktop/PhD/Research/SiDG-ATRID/git/SIDG-ATRID/scripts/Weather/test/snow3.jfif"  # Update with your image path
    
    try:
        weather_class, temperature = predict_weather(image_path, model_path)
        if weather_class is not None:
            print("\nSummary:")
            print(f"Weather Classification: {weather_class}")
            print(f"Temperature Prediction: {temperature:.1f}°C")
        else:
            print("Prediction failed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()