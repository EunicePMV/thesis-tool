import os
import cv2
import numpy as np
from tqdm import tqdm  # Optional, for progress tracking
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Set the path to your dataset directory
dataset_dir = '../balanced-dataset/'

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

# Convert lists to NumPy arrays for further processing
images = np.array(images)
labels = np.array(labels)

# Flatten the images and convert them to a 2D array
images_flat = images.reshape(images.shape[0], -1)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the data
images_scaled = scaler.fit_transform(images_flat)

# Reshape the scaled data back to the original shape
images_scaled = images_scaled.reshape(images.shape)

print("done scaling")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images_scaled, labels, test_size=0.2, random_state=42)

# Reshape the training and testing data
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Initialize the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovo')

# Wrap the SVM classifier in the OneVsOneClassifier
ovo_classifier = OneVsOneClassifier(svm_classifier)

# Fit the classifier on the training data
ovo_classifier.fit(X_train_flat, y_train)

# Make predictions on the test data
y_pred = ovo_classifier.predict(X_test_flat)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", class_report)


# Check the shape of the data
# print("Images shape:", images.shape)
# print("Labels shape:", labels.shape)

# # Display a sample image
# import matplotlib.pyplot as plt
# plt.imshow(images[0])  # Display the first image
# plt.title("Class: " + labels[0])
# plt.show()
