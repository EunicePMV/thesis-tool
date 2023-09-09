import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('./images/result_image.jpg')

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Extract the hue component
hue = hsv_image[:, :, 0]

# Reshape the hue component to prepare it for K-Means clustering
reshaped_hue = hue.reshape((-1, 1))

# Define the number of clusters (2 for diseased and non-diseased parts)
num_clusters = 2

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(reshaped_hue)

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_.squeeze()

# Sort cluster centers by their values to ensure consistency
sorted_centers = np.sort(cluster_centers)

# Determine the threshold value as the middle point between the two clusters
threshold_value = (sorted_centers[0] + sorted_centers[1]) / 2

# Create masks for diseased and non-diseased parts based on the threshold
diseased_mask = hue > threshold_value
non_diseased_mask = hue <= threshold_value

# Set the diseased and non-diseased parts in separate images
diseased_part = np.zeros_like(image)
diseased_part[diseased_mask] = image[diseased_mask]

non_diseased_part = np.zeros_like(image)
non_diseased_part[non_diseased_mask] = image[non_diseased_mask]

# Create histograms for hue values of both parts
hist_diseased = cv2.calcHist([hue], [0], diseased_mask.astype(np.uint8), [256], [0, 256])
hist_non_diseased = cv2.calcHist([hue], [0], non_diseased_mask.astype(np.uint8), [256], [0, 256])

# Display the histograms
plt.subplot(2, 1, 1)
plt.plot(hist_diseased, color='r')
plt.title('Histogram of Hue in Diseased Part')
plt.subplot(2, 1, 2)
plt.plot(hist_non_diseased, color='g')
plt.title('Histogram of Hue in Non-Diseased Part')
plt.show()

# Display the segmented images
diseased_part = cv2.cvtColor(diseased_part, cv2.COLOR_RGB2HSV)
cv2.imshow('Diseased Part', diseased_part)
cv2.imwrite('./images/diseased_part.jpg', diseased_part)
cv2.imshow('Non-Diseased Part', non_diseased_part)
cv2.waitKey(0)
cv2.destroyAllWindows()
