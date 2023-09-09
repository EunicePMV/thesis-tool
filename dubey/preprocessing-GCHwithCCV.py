# ../images/blb_1.jpg
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def compute_ccv(image, segment_mask):
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Blur the LAB image
    blurred_lab_image = cv2.GaussianBlur(lab_image, (5, 5), 0)

    # Extract the L channel
    l_channel = blurred_lab_image[:, :, 0]

    # Find coherent and incoherent pixels
    coherent_pixels = l_channel[segment_mask > 0]
    incoherent_pixels = l_channel[segment_mask == 0]

    # Compute histograms for coherent and incoherent pixels
    coherent_hist = cv2.calcHist([coherent_pixels], [0], None, [32], [0, 256])
    incoherent_hist = cv2.calcHist([incoherent_pixels], [0], None, [32], [0, 256])

    # Concatenate the two histograms
    ccv = np.concatenate((coherent_hist, incoherent_hist), axis=None)

    # Normalize the CCV
    ccv = normalize(ccv.reshape(1, -1), norm='l1')

    return ccv.flatten()

# Read the input image
input_image = cv2.imread('../images/blb_1.jpg')

# Convert the input image to LAB color space
lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)

# Reshape the LAB image to a 2D array of pixels
lab_image_reshaped = lab_image.reshape((-1, 3))

# Define custom lower and upper bounds in the LAB color space
lower_bound = np.array([0, 128, 128])  # Adjust as needed
upper_bound = np.array([255, 255, 255])  # Adjust as needed

# Mask the LAB image based on the custom bounds
mask = cv2.inRange(lab_image, lower_bound, upper_bound)

# Apply k-means clustering on the masked LAB image
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

# Initialize a list to store GCH and CCV features
gch_features = []
ccv_features = []

for i in range(num_clusters):
    segmented_image = np.zeros_like(input_image)
    segmented_image[cluster_labels == i] = input_image[cluster_labels == i]
    segmented_images.append(segmented_image)

    plt.subplot(1, num_clusters + 1, i + 2)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Segment {i + 1}")
    
    # Calculate GCH for each segmented image
    gray_segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    gch_hist = cv2.calcHist([gray_segmented_image], [0], None, [64], [0, 256])
    gch_hist = normalize(gch_hist, norm='l1')
    gch_features.append(gch_hist.flatten())

    # Calculate CCV for each segmented image
    ccv = compute_ccv(segmented_image, cluster_labels == i)
    ccv_features.append(ccv)

# Save the segmented images
for i, segmented_image in enumerate(segmented_images):
    cv2.imwrite(f"segmented_image_{i + 1}.jpg", segmented_image)


# Display GCH and CCV histograms for each segmented image
for i in range(num_clusters):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(segmented_images[i], cv2.COLOR_BGR2RGB))
    plt.title(f"Segment {i + 1}")
    
    plt.subplot(1, 3, 2)
    plt.plot(gch_features[i])
    plt.title(f"GCH for Segment {i + 1}")
    
    plt.subplot(1, 3, 3)
    plt.plot(ccv_features[i])
    plt.title(f"CCV for Segment {i + 1}")
    
plt.show()
