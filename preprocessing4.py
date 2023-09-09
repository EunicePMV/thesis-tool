# inspired with the work of Ramesh and Vydeki

import cv2
import numpy as np

# Load the RGB image
rgb_image = cv2.imread('./images/rice_blast_1.jpg')
rgb_image = cv2.resize(rgb_image, (750, 750))

# Convert RGB to HSV
hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

# Define the lower and upper HSV threshold values for saturation (adjust as needed)
lower_saturation = np.array([30, 0, 0])  # Minimum saturation for thresholding
upper_saturation = np.array([255, 255, 255])  # Maximum saturation for thresholding

# Create a binary mask based on the saturation threshold
saturation_mask = cv2.inRange(hsv_image, lower_saturation, upper_saturation)
# cv2.imshow('Saturated Image', saturation_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Bitwise NOT to create a mask for the background
background_mask = cv2.bitwise_not(saturation_mask)

# Convert the background mask to BGR format for fusion
background_mask_bgr = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

# Apply the mask to the original RGB image to remove the background
result = cv2.bitwise_and(rgb_image, background_mask_bgr)

# Save or display the resulting image with the background removed
cv2.imwrite('rice_leaf_no_background.jpg', result)

# other segmentation: k clustering and fuzzy c 

# STOP HERE: feature extraction
# color: HOG
# texture: LBP and GLCM

