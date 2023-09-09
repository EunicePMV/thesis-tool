# ../images/blb_1.jpg
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


# 1. Read the input image
input_image = cv2.imread('../images/blb_1.jpg')

# 2. Convert the input image to LAB color space
lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)

# 3. Reshape the LAB image to a 2D array of pixels
lab_image_reshaped = lab_image.reshape((-1, 3))

# 4. Define custom lower and upper bounds in the LAB color space
lower_bound = np.array([0, 128, 128])  # Adjust as needed
upper_bound = np.array([255, 255, 255])  # Adjust as needed

# 5. Mask the LAB image based on the custom bounds
mask = cv2.inRange(lab_image, lower_bound, upper_bound)

# 6. Apply k-means clustering on the masked LAB image
masked_lab_image = cv2.bitwise_and(lab_image, lab_image, mask=mask)
masked_lab_image_reshaped = masked_lab_image.reshape((-1, 3))

num_clusters = 2  # Adjust this based on your requirements

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(masked_lab_image_reshaped)
cluster_labels = kmeans.predict(masked_lab_image_reshaped)

# Reshape cluster_labels back to the shape of the original image
cluster_labels = cluster_labels.reshape(lab_image.shape[:2])

# 7. Display the original image and segmented images
plt.subplot(1, num_clusters + 1, 1)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

# Initialize a list to store segmented images
segmented_images = []

# 8. Calculate the Global Color Histogram (GCH) for each segmented image
gch_features = []

for i in range(num_clusters):
    segmented_image = np.zeros_like(input_image)
    segmented_image[cluster_labels == i] = input_image[cluster_labels == i]
    segmented_images.append(segmented_image)
    
    plt.subplot(1, num_clusters + 1, i + 2)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Segment {i + 1}")

# Save the segmented images
for i, segmented_image in enumerate(segmented_images):
    cv2.imwrite(f"segmented_image_{i + 1}.jpg", segmented_image)

plt.show()

for segmented_image in segmented_images:
    # Convert segmented image to grayscale
    gray_segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray_segmented_image], [0], None, [64], [0, 256])
    
    # Normalize the histogram
    hist = normalize(hist, norm='l1')
    
    # Flatten the histogram and append it to the feature list
    gch_features.append(hist.flatten())


# 9. Display the final GCH results
for i, gch_feature in enumerate(gch_features):
    plt.subplot(1, num_clusters, i + 1)
    plt.plot(gch_feature)
    plt.title(f"GCH for Segment {i + 1}")

plt.show()