# High upperTH and lowerTH
import os
import cv2
import numpy as np

# Define the main and output directories
main_directory = "../final-dataset"
output_directory = "../final-dataset-segmented"

# List of classes
classes = ["blb", "rb", "sb", "hlt"]

# Canny edge detection parameters
lower_threshold = 15
upper_threshold = 150

# Mask dilation and erosion parameters
dilation_kernel = np.ones((3, 3), np.uint8)
erosion_kernel = np.ones((3, 3), np.uint8)

# Morphological iterations
MIdilation = 2
MIerosion = 2

# Function to create a mask that fills the entire closed region within the outer contour
def create_convex_polygon_mask(edges):
    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) == 0:
        return np.zeros_like(edges)  # Return an empty mask if no contours are found

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask with the same size as the edges
    mask = np.zeros_like(edges)

    # Fill the largest contour
    cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

    return mask

# Function to perform Canny edge detection and mask dilation/erosion on images in a directory
def apply_edge_detection(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Read the image
            image = cv2.imread(input_path)
            image = cv2.resize(image, (300, 300))

            # Apply Canny edge detection
            edges = cv2.Canny(image, lower_threshold, upper_threshold)

            # Dilation and erosion on edges
            edges = cv2.erode(cv2.dilate(edges, dilation_kernel), erosion_kernel)

            # Create a complete fill mask
            mask = create_convex_polygon_mask(edges)

            # Apply Gaussian blur to the mask
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            # Additional iterations of dilation and erosion
            mask = cv2.dilate(mask, None, iterations=MIdilation)
            mask = cv2.erode(mask, None, iterations=MIerosion)

            # Multiply the mask by 3
            mask_stack = mask * 3

            # Invert the result
            # mask_stack = cv2.bitwise_not(mask)

            # Apply background subtraction
            foreground = cv2.bitwise_and(image, image, mask=mask_stack)

            # Save the result
            cv2.imwrite(output_path, foreground, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


# Iterate through each class and apply edge detection, mask dilation, and mask erosion
for class_name in classes:
    input_dir = os.path.join(main_directory, class_name)
    output_dir = os.path.join(output_directory, class_name)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Apply edge detection, mask dilation, and mask erosion
    apply_edge_detection(input_dir, output_dir)