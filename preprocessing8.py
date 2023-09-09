import cv2
import numpy as np

# Load the RGB image
rgb_image = cv2.imread('./images/rice_blast_1.jpg')
rgb_image = cv2.resize(rgb_image, (750, 750))

# Convert the RGB image to HSV
hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

# Threshold the S channel to create a binary mask
s_threshold = 90  # Adjust this threshold value based on your trials
s_binary = cv2.threshold(hsv_image[:, :, 1], s_threshold, 255, cv2.THRESH_BINARY)[1]

# Create an empty mask with the same dimensions as the RGB image
mask = np.zeros_like(rgb_image)

# Assign the binary mask to all three channels of the mask
mask[:, :, 0] = s_binary
mask[:, :, 1] = s_binary
mask[:, :, 2] = s_binary

# Create a masked image by combining the original RGB image and the mask
masked_image = cv2.bitwise_and(rgb_image, mask)

# Save the resulting image
cv2.imwrite('preprocessing8_image.jpg', masked_image)

# Display the resulting image
cv2.imshow('Background Removed Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
