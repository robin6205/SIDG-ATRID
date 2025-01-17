import os
import json
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. Set Random Seed for Reproducibility
# ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ===============================
# 2. Check Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ===============================
# 3. Define Paths
# ===============================
DATA_DIR = r'D:\SiDG-ATRID-Dataset\Image2Weather\weatherimages\Image'  # Update this path as needed
METADATA_PATH = r'D:\SiDG-ATRID-Dataset\Image2Weather\metadata.json'     # Update this path as needed

# ===============================
# 4. Define Label Classes
# ===============================
label_classes = ['cloudy', 'foggy', 'rain', 'snow', 'sunny']  # Excluding 'z-other'

# ===============================
# 5. Prepare DataFrame with Imputation
# ===============================
def prepare_dataframe(data_dir, metadata_path, label_classes):
    # Load Metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create a dictionary for quick lookup using 'id'
    metadata_dict = {entry['id']: entry for entry in metadata}

    data = []
    skipped_entries = 0

    for label in label_classes:
        label_folder = os.path.join(data_dir, label)
        if not os.path.isdir(label_folder):
            print(f"Warning: Label folder '{label_folder}' does not exist. Skipping this label.")
            continue
        for img_name in os.listdir(label_folder):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue  # Skip non-image files
            img_id = os.path.splitext(img_name)[0]
            entry = metadata_dict.get(img_id, None)
            if entry is None:
                print(f"Warning: No metadata found for image ID '{img_id}'. Skipping.")
                skipped_entries +=1
                continue
            weather_info = entry.get('weather', {})
            tempm = weather_info.get('tempm', None)
            # Check for invalid or missing temperature values
            if tempm in [None, "-9999.0", "-999"]:
                print(f"Warning: Invalid temperature data for image ID '{img_id}'. Skipping.")
                skipped_entries +=1
                continue
            data.append({
                'filename': img_name,
                'label': label,
                'tempm': tempm
            })

    print(f"Total entries parsed (before imputation): {len(data)}")
    print(f"Total entries skipped: {skipped_entries}")

    df = pd.DataFrame(data)

    # Convert 'tempm' to numeric, coercing errors to NaN
    df['tempm'] = pd.to_numeric(df['tempm'], errors='coerce')

    # Verify class distribution
    print("Class distribution:")
    print(df['label'].value_counts())

    # Define classes and count
    classes = sorted(df['label'].unique())
    num_classes = len(classes)
    print(f"Classes: {classes}")

    # Identify NaNs in 'tempm'
    nan_temptm = df['tempm'].isna()
    num_nan_temptm = nan_temptm.sum()
    print(f"Number of NaN 'tempm' entries: {num_nan_temptm}")

    if num_nan_temptm > 0:
        # Initialize imputer (using median strategy)
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')

        # Fit imputer on valid 'tempm' values
        valid_temptm = df.loc[~nan_temptm, ['tempm']]
        imputer.fit(valid_temptm)

        # Replace NaN 'tempm' with imputed values
        df.loc[nan_temptm, 'tempm'] = imputer.transform(df.loc[nan_temptm, ['tempm']])

        print("After imputation, 'tempm' distribution:")
        print(df['tempm'].describe())

        # Verify no NaNs remain in 'tempm'
        remaining_nan_temptm = df['tempm'].isna().sum()
        print(f"Number of remaining NaN 'tempm' entries after imputation: {remaining_nan_temptm}")

        if remaining_nan_temptm > 0:
            print("Warning: There are still NaN values in 'tempm' after imputation. Applying additional imputation.")
            # Apply additional imputation if necessary
            df['tempm'] = imputer.fit_transform(df[['tempm']])

        # Final check
        if df['tempm'].isna().sum() == 0:
            print("All 'tempm' values have been successfully imputed.")
        else:
            print("Some 'tempm' values remain as NaN. Please check the imputation step.")
    else:
        print("No NaN 'tempm' entries to impute.")

    # Final verification
    total_nans = df['tempm'].isna().sum()
    print(f"Total NaN values in 'tempm' after imputation: {total_nans}")

    if total_nans > 0:
        print("Warning: There are still NaN values in the temperature targets. Please handle them accordingly.")
    else:
        print("All temperature targets are free of NaN values.")

    return df, classes, num_classes

# ===============================
# 6. Define Custom Dataset
# ===============================
class WeatherDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None, class_to_idx=None, scaler=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.scaler = scaler  # StandardScaler instance for temperature

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'filename']
        label_name = self.dataframe.loc[idx, 'label']
        label = self.class_to_idx[label_name]

        tempm = self.dataframe.loc[idx, 'tempm']

        # Image path
        img_path = os.path.join(self.data_dir, label_name, img_name)
        if not os.path.exists(img_path):
            # Handle missing images gracefully
            print(f"Warning: Image not found at '{img_path}'. Skipping.")
            # You can choose to skip or return a default image
            # Here, we'll return a tensor of zeros
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Normalize temperature
        # if self.scaler:
        #     tempm = self.scaler.transform([[tempm]])[0][0]
        if self.scaler:
            tempm = self.scaler.transform(pd.DataFrame({'tempm': [tempm]}))[0][0]

        # Regression target
        regression_target = torch.tensor([tempm], dtype=torch.float32)

        return image, label, regression_target

# ===============================
# 7. Define the Multi-Task Model
# ===============================
class MultiTaskResNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskResNet, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze the last block
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

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
            nn.Linear(256, 1)  # Predicting temperature
        )

        # Initialize regression head
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        classification = self.classifier(features)
        regression = self.regressor(features)
        return classification, regression

# ===============================
# 8. Training Function
# ===============================
def train_model_multitask(model, classification_criterion, regression_criterion, optimizer, scheduler, 
                         train_loader, val_loader, num_epochs=25, patience=5):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    train_class_acc = []
    val_class_acc = []
    train_reg_mae = []
    val_reg_mae = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # For regression metrics
            all_preds_reg = []
            all_targets_reg = []

            # Iterate over data
            for batch in tqdm(dataloader, desc=phase):
                if batch is None:
                    continue  # Skip batches with no data
                inputs, labels, regressions = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                regressions = regressions.to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_class, outputs_reg = model(inputs)

                    # Check for NaN in outputs
                    if torch.isnan(outputs_class).any():
                        print("Warning: NaN detected in classification outputs!")
                    if torch.isnan(outputs_reg).any():
                        print("Warning: NaN detected in regression outputs!")

                    loss_class = classification_criterion(outputs_class, labels)
                    loss_reg = regression_criterion(outputs_reg, regressions)
                    loss = loss_class + loss_reg

                    if phase == 'train':
                        loss.backward()
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs_class, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                # Collect regression predictions and targets
                all_preds_reg.append(outputs_reg.detach().cpu().numpy())
                all_targets_reg.append(regressions.detach().cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            # Concatenate all regression predictions and targets
            all_preds_reg = np.concatenate(all_preds_reg, axis=0)
            all_targets_reg = np.concatenate(all_targets_reg, axis=0)
            epoch_reg_mae = mean_absolute_error(all_targets_reg, all_preds_reg)
            epoch_reg_rmse = np.sqrt(mean_squared_error(all_targets_reg, all_preds_reg))

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_class_acc.append(epoch_acc.item())
                train_reg_mae.append(epoch_reg_mae)
            else:
                val_losses.append(epoch_loss)
                val_class_acc.append(epoch_acc.item())
                val_reg_mae.append(epoch_reg_mae)

            print(f'{phase} Loss: {epoch_loss:.4f} | Class Acc: {epoch_acc:.4f} | Reg MAE: {epoch_reg_mae:.4f} | Reg RMSE: {epoch_reg_rmse:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                epochs_no_improve = 0
            elif phase == 'val':
                epochs_no_improve +=1
                print(f'EarlyStopping counter: {epochs_no_improve} out of {patience}')
                if epochs_no_improve >= patience:
                    print('Early stopping!')
                    model.load_state_dict(best_model_wts)
                    return model

        scheduler.step()
        print()

    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot training and validation loss
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    # Plot training and validation classification accuracy
    plt.subplot(1,2,2)
    plt.plot(range(1, len(train_class_acc)+1), train_class_acc, label='Train Class Acc')
    plt.plot(range(1, len(val_class_acc)+1), val_class_acc, label='Val Class Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Classification Accuracy over Epochs')

    plt.tight_layout()
    plt.show()

    # Plot regression MAE and RMSE
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, len(train_reg_mae)+1), train_reg_mae, label='Train Reg MAE')
    plt.plot(range(1, len(val_reg_mae)+1), val_reg_mae, label='Val Reg MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Regression MAE over Epochs')

    plt.subplot(1,2,2)
    plt.plot(range(1, len(train_reg_mae)+1), train_reg_mae, label='Train Reg MAE')
    plt.plot(range(1, len(val_reg_mae)+1), val_reg_mae, label='Val Reg MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Regression MAE over Epochs')

    plt.tight_layout()
    plt.show()

    return model

# ===============================
# 9. Evaluation Function
# ===============================
def evaluate_model_multitask(model, dataloader, device, idx_to_class, classes):
    model.eval()
    all_preds_class = []
    all_labels_class = []
    all_preds_reg = []
    all_targets_reg = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Test'):
            if batch is None:
                continue  # Skip empty batches
            inputs, labels, regressions = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs_class, outputs_reg = model(inputs)
            _, preds_class = torch.max(outputs_class, 1)

            all_preds_class.extend(preds_class.cpu().numpy())
            all_labels_class.extend(labels.cpu().numpy())

            all_preds_reg.append(outputs_reg.cpu().numpy())
            all_targets_reg.append(regressions.cpu().numpy())

    # Classification Metrics
    print("Classification Report:")
    print(classification_report(all_labels_class, all_preds_class, target_names=classes))

    # Confusion Matrix
    cm = confusion_matrix(all_labels_class, all_preds_class)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # Regression Metrics
    all_preds_reg = np.concatenate(all_preds_reg, axis=0).flatten()
    all_targets_reg = np.concatenate(all_targets_reg, axis=0).flatten()

    mae = mean_absolute_error(all_targets_reg, all_preds_reg)
    rmse = np.sqrt(mean_squared_error(all_targets_reg, all_preds_reg))

    print(f"Regression Metrics:")
    print(f"Temperature (C) - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Plot predicted vs actual for regression
    plt.figure(figsize=(8,6))
    plt.scatter(all_targets_reg, all_preds_reg, alpha=0.5)
    plt.plot([all_targets_reg.min(), all_targets_reg.max()], [all_targets_reg.min(), all_targets_reg.max()], 'r--')
    plt.xlabel('Actual Temperature (C)')
    plt.ylabel('Predicted Temperature (C)')
    plt.title('Actual vs Predicted Temperature')
    plt.show()

# ===============================
# 10. Main Function
# ===============================
def main():
    # Prepare DataFrame
    df, classes, num_classes = prepare_dataframe(DATA_DIR, METADATA_PATH, label_classes)

    # Train-Validation-Test Split
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # Mapping from class names to indices
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

    # Initialize Scaler for Temperature
    scaler_temptm = StandardScaler()

    # Fit scaler on training data
    scaler_temptm.fit(train_df[['tempm']])

    # Define Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Mean for ImageNet
                             [0.229, 0.224, 0.225])  # Std for ImageNet
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Create Datasets with class_to_idx and scaler
    train_dataset = WeatherDataset(train_df, DATA_DIR, transform=train_transforms, class_to_idx=class_to_idx, scaler=scaler_temptm)
    val_dataset = WeatherDataset(val_df, DATA_DIR, transform=val_test_transforms, class_to_idx=class_to_idx, scaler=scaler_temptm)
    test_dataset = WeatherDataset(test_df, DATA_DIR, transform=val_test_transforms, class_to_idx=class_to_idx, scaler=scaler_temptm)

    # Create DataLoaders
    batch_size = 32
    num_workers = 4  # Adjust based on your system

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize the model
    model = MultiTaskResNet(num_classes=num_classes)
    model = model.to(device)

    # Define Loss Functions
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

    # Define Optimizer with a lower learning rate
    optimizer = torch.optim.Adam([
        {'params': model.classifier.parameters()},
        {'params': model.regressor.parameters()}
    ], lr=1e-4)  # Reduced learning rate from 1e-3 to 1e-4

    # Define Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the Model
    num_epochs = 25
    patience = 5  # Early stopping patience

    trained_model = train_model_multitask(
        model, 
        classification_criterion, 
        regression_criterion, 
        optimizer, 
        scheduler, 
        train_loader, 
        val_loader, 
        num_epochs=num_epochs, 
        patience=patience
    )

    # Save the Model
    model_save_path = 'weather_classifier_regressor_resnet50.pth'
    torch.save(trained_model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Evaluation on Test Set
    evaluate_model_multitask(trained_model, test_loader, device, idx_to_class, classes)

if __name__ == '__main__':
    main()
