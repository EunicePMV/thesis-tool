import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
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

# Assuming you have already applied background removal and have two images: 
# 'blast_affected' (containing the diseased portion) and 'normal' (containing the normal portion)

# Convert the background-removed images to HSV color space
hsv_blast_affected = cv2.cvtColor(blast_affected, cv2.COLOR_BGR2HSV)
hsv_normal = cv2.cvtColor(normal, cv2.COLOR_BGR2HSV)

# Extract the hue component for both background-removed images
hue_blast_affected = hsv_blast_affected[:, :, 0]
hue_normal = hsv_normal[:, :, 0]

# Display the histogram of the hue component of the background-removed diseased portion
plt.subplot(132)
plt.hist(hue_blast_affected.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
plt.title('Histogram of Hue (Blast Affected)')
plt.xlabel('Hue Value')
plt.ylabel('Frequency')

# Display the histogram of the hue component of the background-removed normal portion
plt.subplot(133)
plt.hist(hue_normal.ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
plt.title('Histogram of Hue (Normal)')
plt.xlabel('Hue Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
