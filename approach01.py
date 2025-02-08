import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.filters import frangi
from PIL import Image

# -------------------------------
# 1️⃣ Load and Preprocess Images
# -------------------------------

class SkinLesionDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.labels = [self.get_label(f) for f in self.image_files]  # Extract labels from filenames

    def get_label(self, filename):
        # Example: "melanoma_001.jpg" -> Label = "melanoma"
        return filename.split("_")[0]

    def frangi_filter(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Apply Frangi vessel enhancement filter
        vessels = frangi(gray)
        # Normalize to 0-255
        vessels = (255 * (vessels - vessels.min()) / (vessels.max() - vessels.min())).astype(np.uint8)
        return vessels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Frangi filter
        vessels = self.frangi_filter(image)

        # Convert images to PIL format for transformation
        image = Image.fromarray(image)
        vessels = Image.fromarray(vessels)

        if self.transform:
            image = self.transform(image)
            vessels = self.transform(vessels)

        return torch.cat((image, vessels), dim=0), self.labels[idx]  # Stack original + vessel image

# -------------------------------
# 2️⃣ Define CNN Model
# -------------------------------

class CNN_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Model, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)  # 4 channels (RGB + Vessel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------
# 3️⃣ Training & Evaluation
# -------------------------------

def train_model(model, train_loader, num_classes, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = torch.tensor([class_to_idx[lbl] for lbl in labels]).to(device)  # Encode labels

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

    return model

# -------------------------------
# 4️⃣ Main Execution
# -------------------------------

if __name__ == "__main__":
    img_dir = "/path/to/skin_lesions/"  # Update with your dataset path

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = SkinLesionDataset(img_dir, transform=transform)
    
    # Encode class labels
    class_names = list(set(dataset.labels))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    num_classes = len(class_names)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CNN_Model(num_classes=num_classes)
    model = train_model(model, train_loader, num_classes)
