import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the background-removed image
background_removed_image = cv2.imread('clustered_image.jpg')

# Convert the image to HSV color space
image_hsv = cv2.cvtColor(background_removed_image, cv2.COLOR_BGR2HSV)

# Extract the hue channel
hue_channel = image_hsv[:, :, 0]

# Create a histogram of hue values
histogram, bins = np.histogram(hue_channel, bins=256, range=(0, 256))

# Find the bin with the maximum count as a potential threshold
max_bin = np.argmax(histogram)

# Use a simple thresholding technique (e.g., using Otsu's method)
_, thresholded = cv2.threshold(hue_channel, max_bin, 255, cv2.THRESH_BINARY)

# Store hue values in separate arrays for diseased and normal portions
hue_values_diseased = hue_channel[thresholded == 255]
hue_values_normal = hue_channel[thresholded == 0]

# Display the histogram
plt.figure(figsize=(8, 4))
plt.hist(hue_channel.flatten(), bins=256, range=(0, 256), color='b', alpha=0.7)
plt.axvline(x=max_bin, color='r', linestyle='--', linewidth=2, label='Threshold')
plt.title('Hue Histogram')
plt.xlabel('Hue Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
