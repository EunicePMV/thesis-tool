import cv2
import numpy as np

# Load the image
image = cv2.imread('./dataset-balanced/bacterial-leaf-blight/a.jpg')

# Preprocess the image if necessary (resize, denoise, adjust contrast, etc.)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Sobel operator
sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)

# Thresholding
thresholded = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)[1]

# Find contours
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and select the contour representing the leaf (you may need to adjust criteria)
selected_contour = None  # Replace with your logic to select the contour

# Create a mask for the selected contour
mask = cv2.drawContours(np.zeros_like(image), [selected_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Extract the leaf using the mask
extracted_leaf = cv2.bitwise_and(image, mask)

# Display the extracted leaf image
cv2.imshow('Extracted Leaf', extracted_leaf)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
