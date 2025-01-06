# Import necessary libraries
import cv2
import os
import numpy as np
from albumentations import Compose, RandomBrightnessContrast, Flip, ElasticTransform, GridDistortion, HueSaturationValue, CLAHE, RandomRotate90
from albumentations import HorizontalFlip, VerticalFlip

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define paths
data_path = '/content/drive/MyDrive/B.E. Project/archive 2/train_path/'
output_path = '/content/drive/MyDrive/B.E. Project/archive 2/newly_augmented_2'

# Augmentation pipeline
augmentations = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.3),
    GridDistortion(p=0.3),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    CLAHE(clip_limit=2.0, p=0.3),
    RandomRotate90(p=0.5),
])

# Debugging functions
def log_image_info(stage, img):
    print(f"[{stage}] Shape: {img.shape}, Dtype: {img.dtype}, Range: [{img.min()} - {img.max()}]")

def save_debug_image(output_path, img, stage):
    if img.max() <= 1.0:  # If normalized
        img = (img * 255).astype('uint8')
    elif img.dtype != 'uint8':  # If dtype is not uint8
        img = img.astype('uint8')
    cv2.imwrite(output_path, img)
    print(f"[DEBUG] Saved image at {stage}: {output_path}")

# Hair removal function
def remove_hair(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        inpainted = cv2.inpaint(img, threshold, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return inpainted
    except Exception as e:
        print(f"Error in hair removal: {e}")
        return img  # Return the original image if hair removal fails

# Preprocessing function
def preprocess_images(data_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    classes = os.listdir(data_path)

    for class_name in classes:
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"Processing class: {class_name}")
        output_class_dir = os.path.join(output_path, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error loading image {img_name}. Skipping.")
                continue

            # Log original image properties
            log_image_info("Original Image", img)

            # Hair removal
            img_no_hair = remove_hair(img)
            log_image_info("After Hair Removal", img_no_hair)

            # Save debug image
            save_debug_image(os.path.join(output_class_dir, f"debug_no_hair_{img_name}"), img_no_hair, "Hair Removal")

            # Augmentation
            for i in range(3):  # Generate 3 augmented images
                augmented = augmentations(image=img_no_hair)['image']
                log_image_info(f"Augmented Image {i}", augmented)

                # Save augmented image
                output_img_path = os.path.join(output_class_dir, f"aug_{i}_{img_name}")
                save_debug_image(output_img_path, augmented, f"Augmentation {i}")

# Run preprocessing
preprocess_images(data_path, output_path)
