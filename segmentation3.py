import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original leaf image
image = cv2.imread('clustered_image.jpg')

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Extract the hue component
hue = hsv_image[:, :, 0]

# Display the histogram of the hue component of the entire image
plt.figure(figsize=(8, 4))
plt.subplot(131)
plt.hist(hue.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
plt.title('Histogram of Hue (Entire Image)')
plt.xlabel('Hue Value')
plt.ylabel('Frequency')

# Perform thresholding to segment the blast-affected portion
# You need to determine an appropriate threshold value based on your image
# For example, you can use a simple fixed threshold:
threshold_value = 120  # Adjust this value as needed
blast_affected_mask = (hue < threshold_value).astype(np.uint8)

# Apply the mask to the original image to get the blast-affected portion
blast_affected = cv2.bitwise_and(image, image, mask=blast_affected_mask)

# Display the histogram of the hue component of the blast-affected portion
plt.subplot(132)
plt.hist(hue[blast_affected_mask == 1].ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
plt.title('Histogram of Hue (Blast Affected)')
plt.xlabel('Hue Value')
plt.ylabel('Frequency')

# Invert the mask to get the normal portion
normal_mask = 255 - blast_affected_mask

# Apply the inverted mask to the original image to get the normal portion
normal = cv2.bitwise_and(image, image, mask=normal_mask)

# Display the histogram of the hue component of the normal portion
plt.subplot(133)
plt.hist(hue[normal_mask == 1].ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
plt.title('Histogram of Hue (Normal)')
plt.xlabel('Hue Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
