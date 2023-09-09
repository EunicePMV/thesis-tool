import cv2
import numpy as np

# Load the original RGB image
original_image = cv2.imread('../images/K-blb_3.jpg')
original_image = cv2.resize(original_image, (300, 450))
cv2.imshow('Original Image', original_image)


# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image', hsv_image)

# Extract the Saturation (S) channel
# s_channel = hsv_image[:, :, 1]

# TRIAL
lower_saturation = np.array([0, 90, 0])  # Minimum saturation for thresholding
upper_saturation = np.array([255, 255, 255])  # Maximum saturation for thresholding

saturation_mask = cv2.inRange(hsv_image, lower_saturation, upper_saturation)

# cv2.imshow('Saturation Part', saturation_mask)


# Apply a threshold to create a binary mask
threshold_value = 90
binary_mask = cv2.threshold(saturation_mask, threshold_value, 30, cv2.THRESH_BINARY)[1]

cv2.imshow('Binary Image', binary_mask)

# Create an all-black image of the same size as the original
black_background = np.zeros_like(original_image)

# Use the binary mask to copy the leaf portion from the original image to the black background
result_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask)

# Save the result
cv2.imwrite('./images/result_image.jpg', result_image)

# Display the result (optional)
cv2.imshow('Result Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()