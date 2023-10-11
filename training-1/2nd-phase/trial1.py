# 1. for each image apply preprocesssing (what are the steps to preprocess images)
# 2. load annotated and preprocessed images
# 3. segment
# 4. feature extraction
# 5. normalize dataset
# 6. how to train with the extracted features

# AFTER SUCCESSFUL SEGMENTAION, TRY THE SAME PROCEDURE TO PHILRICE DATA

import os
import cv2
import numpy as np
from tqdm import tqdm  # Optional, for progress tracking
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Set the path to your dataset directory
dataset_dir = './dataset-color_enhancement/'

# Initialize empty lists to store images and labels
images = []
labels = []

# Iterate through each folder (class) in the dataset directory
for class_folder in os.listdir(dataset_dir):
    if os.path.isdir(os.path.join(dataset_dir, class_folder)):
        class_label = class_folder

        # Iterate through each image in the class folder
        for image_filename in tqdm(os.listdir(os.path.join(dataset_dir, class_folder))):
            image_path = os.path.join(dataset_dir, class_folder, image_filename)

            # Load the image using OpenCV
            image = cv2.imread(image_path)
            
            # Resize the image (optional)
            image = cv2.resize(image, (224, 224))

            # Preprocess the image (e.g., normalize pixel values)
            # image = image / 255.0  # Normalize pixel values to [0, 1]

            # Append the image and label to the lists
            images.append(image)
            labels.append(class_label)

# ======================= NORMALIZE DATASET =============================
# Convert lists to NumPy arrays for further processing
images = np.array(images)
labels = np.array(labels)

# Flatten the images and convert them to a 2D array
images_flat = images.reshape(images.shape[0], -1)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the data
images_scaled = scaler.fit_transform(images_flat)
