import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # For the progress bar

# Path to the main folder
main_folder = "/content/drive/MyDrive/B.E. Project/archive 2/extracted_imgs/"

# List sub-folders (classes)
sub_folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

# Initialize a color palette for each class
colors = plt.cm.rainbow(np.linspace(0, 1, len(sub_folders)))

# To store min and max values for consistent axes across all plots
min_x, max_x = float('inf'), float('-inf')
min_y, max_y = float('inf'), float('-inf')

# Process each sub-folder (class) with a progress bar
for i, sub_folder in enumerate(tqdm(sub_folders, desc="Processing Classes")):
    folder_path = os.path.join(main_folder, sub_folder)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Combine features and labels from all CSV files in the sub-folder
    features = []
    labels = []
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(csv_path)

        # Assume the last column is the class label
        features.append(data.iloc[:, :-1])
        labels.append(data.iloc[:, -1])

    # Concatenate data from all CSVs in the sub-folder
    features = pd.concat(features, ignore_index=True)
    labels = pd.concat(labels, ignore_index=True)

    # Standardize the features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform PCA on the scaled features
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)

    # Find the min and max values for PCA results across all classes for consistent axis limits
    min_x = min(min_x, pca_result[:, 0].min())
    max_x = max(max_x, pca_result[:, 0].max())
    min_y = min(min_y, pca_result[:, 1].min())
    max_y = max(max_y, pca_result[:, 1].max())

# Now plot for each class with consistent axis limits
for i, sub_folder in enumerate(tqdm(sub_folders, desc="Plotting Classes")):
    folder_path = os.path.join(main_folder, sub_folder)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Combine features and labels from all CSV files in the sub-folder
    features = []
    labels = []
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(csv_path)

        # Assume the last column is the class label
        features.append(data.iloc[:, :-1])
        labels.append(data.iloc[:, -1])

    # Concatenate data from all CSVs in the sub-folder
    features = pd.concat(features, ignore_index=True)
    labels = pd.concat(labels, ignore_index=True)

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform PCA on the scaled features
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)

    # Create a new plot for this subclass (class)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], label=sub_folder, alpha=0.7, color=colors[i])
    
    # Set consistent axis limits for all plots
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # Customize plot for this subclass
    plt.xlabel("Principal Component 1 (Best Feature 1)")
    plt.ylabel("Principal Component 2 (Best Feature 2)")
    plt.title(f"PCA Visualization of {sub_folder} Class")
    plt.legend()
    plt.grid(True)
    plt.show()
