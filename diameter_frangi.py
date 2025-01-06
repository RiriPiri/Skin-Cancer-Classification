import os
import cv2
import numpy as np
from skimage.filters import frangi
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Function to apply Frangi filter and save the output for all images in a folder
def extract_blood_vessels_from_folder(folder_path, output_folder, diameters=(3, 5, 7)):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Process only image files
            image_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            try:
                # Read the image
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error: Unable to read image {image_path}")
                    continue  # Skip this image and continue with others
                print(f"Image shape: {img.shape}")

                # Convert to grayscale
                gray_img = rgb2gray(img)
                print(f"Converted to grayscale. Shape: {gray_img.shape}")

                # Apply Frangi filter to extract blood vessels with specific diameters (sigmas)
                vessels = frangi(gray_img, sigmas=diameters)
                print(f"Frangi filter applied. Shape: {vessels.shape}")

                # Normalize the output for better visualization
                vessels_normalized = (vessels - vessels.min()) / (vessels.max() - vessels.min())

                # Save the result
                output_path = os.path.join(output_folder, f"vessels_{filename}")
                plt.imsave(output_path, vessels_normalized, cmap='gray')
                print(f"Blood vessel image saved to: {output_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Test with a folder
#folder_path = "/content/drive/MyDrive/B.E. Project/archive 2/newly_augmented_2/"  # Replace with the path to your folder containing images
#output_folder = "/content/drive/MyDrive/B.E. Project/archive 2/extractions_2/"  # Replace with the folder to save output images

# Run the processing for the entire folder with specified vessel diameters
#extract_blood_vessels_from_folder(folder_path, output_folder, diameters=(3, 5, 7))

extract_blood_vessels("/content/drive/MyDrive/B.E. Project/archive 2/newly_augmented_2/BKL/debug_no_hair_ISIC_0026590.jpg", "/content/drive/MyDrive/B.E. Project/archive 2/extractions_2/")
