import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

# Custom Dataset for loading images and precomputed vessel images
class SkinCancerDataset(Dataset):
    def __init__(self, image_paths, vessel_paths, labels, transform=None):
        self.image_paths = image_paths
        self.vessel_paths = vessel_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        vessel_path = self.vessel_paths[idx]
        label = self.labels[idx]

        # Load original image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Load vessel image (already precomputed)
        vessels = cv2.imread(vessel_path, cv2.IMREAD_GRAYSCALE)
        vessels = Image.fromarray(vessels).convert("RGB")  # Convert grayscale to 3-channel

        if self.transform:
            image = self.transform(image)
            vessels = self.transform(vessels)

        # Concatenate original image and vessel image along the channel dimension
        combined_input = torch.cat((image, vessels), dim=0)  # Shape: (6, 224, 224)

        return combined_input, label

# CNN Model
class SkinCancerCNN(nn.Module):
    def __init__(self, num_classes):
        super(SkinCancerCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to load dataset paths
def load_data(image_dir, vessel_dir):
    classes = sorted(os.listdir(image_dir))  # Folder names are labels
    image_paths, vessel_paths, labels = [], [], []
    label_map = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        img_cls_path = os.path.join(image_dir, cls)
        vessel_cls_path = os.path.join(vessel_dir, cls)  # Vessel images stored separately

        for img in os.listdir(img_cls_path):
            img_path = os.path.join(img_cls_path, img)
            vessel_path = os.path.join(vessel_cls_path, img)  # Matching filename

            if os.path.exists(vessel_path):  # Ensure vessel image exists
                image_paths.append(img_path)
                vessel_paths.append(vessel_path)
                labels.append(label_map[cls])

    return train_test_split(image_paths, vessel_paths, labels, test_size=0.2, stratify=labels, random_state=42), len(classes)

# Train Model
def train_model(image_dir, vessel_dir, epochs=10, batch_size=16):
    (train_imgs, test_imgs, train_vessels, test_vessels, train_labels, test_labels), num_classes = load_data(image_dir, vessel_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = SkinCancerDataset(train_imgs, train_vessels, train_labels, transform)
    test_dataset = SkinCancerDataset(test_imgs, test_vessels, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SkinCancerCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    return model, test_loader

# Evaluate Model
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
