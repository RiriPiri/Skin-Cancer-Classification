import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Define the CNN Model
class CNN_Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN_Model, self).__init__()
        
        # Convolutional layer: input channels=1, output channels=32, kernel size=3
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the size of the output after convolution and pooling
        conv_output_size = (input_dim - 2) // 2  # 2 is for the kernel size (3) and pooling (kernel_size=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * conv_output_size, 128)  # Adjusted based on conv output size
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Apply convolution and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        
        # Flatten the tensor before passing it to fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU activation
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer
        return x

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Drop unwanted columns
    data = data.drop(['set_type', 'image_name'], axis=1)

    # Encode target variable
    label_encoder = LabelEncoder()
    data['folder_name'] = label_encoder.fit_transform(data['folder_name'])

    # Split features and target
    X = data.drop('folder_name', axis=1)
    y = data['folder_name']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return pd.DataFrame(X_train), pd.DataFrame(X_test), y_train, y_test, len(label_encoder.classes_)

# Train CNN
def train_cnn(X_train, y_train, X_test, y_test, num_classes, epochs=50, batch_size=32):
    # Prepare dataset
    X_train['target'] = y_train
    X_test['target'] = y_test
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float).unsqueeze(1)  # Add channel dimension
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    # Ensure the model uses the CPU
    device = torch.device('cpu')

    # Initialize CNN model
    model = CNN_Model(input_dim=X_train.shape[1], num_classes=num_classes)
    model.to(device)  # Ensure the model uses the CPU

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model, X_test_tensor, y_test_tensor

# Evaluation
def evaluate_model(model, X_test, y_test, num_classes):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        _, predicted = torch.max(y_pred, 1)

    # AUC
    y_pred_proba = torch.nn.functional.softmax(y_pred, dim=1).cpu().numpy()
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    print(f"AUC Score: {auc:.4f}")

    # Classification Report
    predicted = predicted.cpu().numpy()
    print("\nClassification Report:")
    print(classification_report(y_test, predicted))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predicted)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return auc

# Main function
def main():
    # Load data
    file_path = "/content/PCA_test.csv"  # Replace with your file path
    X_train, X_test, y_train, y_test, num_classes = load_data(file_path)

    # Train CNN Model
    model, X_test_tensor, y_test_tensor = train_cnn(X_train, y_train, X_test, y_test, num_classes, epochs=50)

    # Evaluate the model
    auc = evaluate_model(model, X_test_tensor, y_test_tensor, num_classes)
    print(f"Final AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
