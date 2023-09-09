import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('preprocessing6_image.jpg')

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Extract the hue component (0-179 in OpenCV)
hue = hsv_image[:,:,0]

# Flatten the hue component for clustering
hue_flat = hue.reshape((-1, 1))

# Define the number of clusters (2 for diseased and non-diseased parts)
num_clusters = 2

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(hue_flat)

# Get the cluster labels and reshape them back to the image shape
cluster_labels = kmeans.labels_.reshape(hue.shape)

# Calculate the centroids of the clusters
centroids = kmeans.cluster_centers_

# Display the segmented image
segmented_image = np.zeros_like(hue)
for i in range(num_clusters):
    segmented_image[cluster_labels == i] = centroids[i]

# Display the segmented image
plt.imshow(segmented_image, cmap='hsv')
plt.title('Segmented Image (Hue)')
plt.axis('off')
plt.show()

# Create a histogram of the hue component
hist = cv2.calcHist([hue], [0], None, [180], [0, 180])

# Display the histogram
plt.figure()
plt.plot(hist)
plt.title('Hue Histogram')
plt.xlabel('Hue Value')
plt.ylabel('Frequency')
plt.xlim([0, 180])
plt.show()
