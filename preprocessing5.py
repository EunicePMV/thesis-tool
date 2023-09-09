# k-means clustering

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('./images/rice_blast_1.jpg')
image = cv2.resize(image, (300, 450))

# Convert the image from BGR to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the saturation channel
lower_saturation = np.array([0, 0, 0])
upper_saturation = np.array([255, 90, 255])

# Create a binary mask where the pixels with saturation values within the threshold become white (255), and others black (0)
mask = cv2.inRange(hsv_image, lower_saturation, upper_saturation)

# Apply morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Invert the mask
mask = cv2.bitwise_not(mask)

# Create a masked image by applying the mask
masked_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite('result.jpg', masked_image)

# Load the background-removed image (previously obtained)
removed_background_image = cv2.imread('./result.jpg')

# Convert the image from BGR to HSV color space
hsv_removed_background = cv2.cvtColor(removed_background_image, cv2.COLOR_BGR2HSV)

# Extract the hue component (channel 0)
hue_component = hsv_removed_background[:, :, 0]

# Flatten the hue component array to prepare it for k-means clustering
hue_flat = hue_component.reshape((-1, 1))

# Define the number of clusters (k) for k-means
k = 5  # You can adjust this value based on your needs

# Apply k-means clustering
kmeans = KMeans(n_clusters=k)
kmeans.fit(hue_flat)

# Get the cluster labels for each pixel
labels = kmeans.predict(hue_flat)

# Reshape the labels back to the original shape
labels = labels.reshape(hue_component.shape)

# Create a mask for each cluster label
masks = [labels == i for i in range(k)]

# Create a blank image for visualization
clustered_image = np.zeros_like(removed_background_image)

# Assign colors to each cluster based on the label
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]  # You can adjust these colors
for i, mask in enumerate(masks):
    clustered_image[mask] = colors[i]

# Display or save the clustered image
cv2.imshow('Clustered Image', clustered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# If you want to save the result
cv2.imwrite('clustered_image.jpg', clustered_image)