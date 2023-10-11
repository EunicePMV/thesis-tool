# NOTE: MANUAL PROCESS TO NORMALIZE

import numpy as np
from sklearn.preprocessing import StandardScaler
import cv2  # You might need to install OpenCV: pip install opencv-python
import os

# Assuming you have a list of image file paths in 'image_paths'
# And you have specified the size of your images in 'image_size'
image_directory = './dataset/bacterial-leaf-blight/train'
image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith('.jpg')]

# image_size = (64, 64) 

# Load and preprocess images
images = []
for image_path in image_paths:
    img = cv2.imread(image_path)  # Load the image
    # img = cv2.resize(img, image_size)  # Resize the image to your desired size
    images.append(img)

# Convert the list of images to a NumPy array
X_train = np.array(images)

# Calculate the mean and standard deviation of the training dataset
mean = np.mean(X_train, axis=(0, 1, 2))
std = np.std(X_train, axis=(0, 1, 2))

# Avoid division by zero, add a small epsilon if std is too close to zero
epsilon = 1e-8
std += epsilon

# Normalize the training dataset
X_train_normalized = (X_train - mean) / std

# The mean of X_train_normalized should now be very close to zero, and std should be close to 1.

# Save the mean and std values for later use during inference or for normalizing the validation and test datasets
np.save('mean.npy', mean)
np.save('std.npy', std)

                    # TO NORMALIZE VALIDATION AND TEST DATASET
                    # Load the mean and std values
                    # mean = np.load('mean.npy')
                    # std = np.load('std.npy')

                    # Normalize the validation and test datasets
                    # X_val_normalized = (X_val - mean) / std
                    # X_test_normalized = (X_test - mean) / std

# Now you can use X_train_normalized for training your model.
