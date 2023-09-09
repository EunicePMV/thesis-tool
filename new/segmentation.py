import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the RGB image
rgb_image = cv2.imread('../images/blb_1.jpg')  # Replace with the path to your RGB image

# Convert RGB to BGR (OpenCV uses BGR by default)
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

# Convert BGR to HSI
hsi_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

# Reshape the HSI image into a 2D array
hsi_pixels = hsi_image.reshape((-1, 3))

# Specify the number of clusters (K)
k = 2  # You can change this value to the desired number of clusters

# Create a K-means clustering model
kmeans = KMeans(n_clusters=k)

# Fit the model to the HSI data
kmeans.fit(hsi_pixels)

# Get the cluster labels for each pixel
cluster_labels = kmeans.labels_

# Reshape the cluster labels to the original image shape
clustered_image = cluster_labels.reshape(hsi_image.shape[:2])

# Save or display the clustered HSI image
cv2.imwrite('clustered_image.jpg', clustered_image.astype(np.uint8))  # Save the image as JPEG
