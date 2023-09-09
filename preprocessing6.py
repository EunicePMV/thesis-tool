import cv2
import numpy as np

# Load the image in RGB format
image = cv2.imread('./images/blb_1.jpg')
image = cv2.resize(image, (750, 750))

# Convert the image from BGR to RGB (if it's in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the RGB image to HSV
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# Threshold the S channel to create a binary mask
threshold_value = 90
_, mask = cv2.threshold(image_hsv[:, :, 1], threshold_value, 250, cv2.THRESH_BINARY)

# Create an inverted mask to keep the foreground
mask_inv = cv2.bitwise_not(mask)

# Apply the mask to eliminate the background
result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Convert the result back to BGR format if needed
result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

# Display the original image and the background-removed image
cv2.imshow('Original Image', image)
cv2.imshow('Background Removed Image', result_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the background-removed image
cv2.imwrite('preprocessing6_image.jpg', result_bgr)

