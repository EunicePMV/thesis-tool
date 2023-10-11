import cv2
import os
import numpy as np

# Define the directory where your input images are stored
input_directory = './dataset-bilateralFil'

# Define the directory where you want to save the processed images
output_directory = './edge-extraction/13'

# Create a list of class names
classes = ['bacterial-leaf-blight', 'rice-blast', 'rice-tungro', 'healthy']

# Function to determine if a contour meets the criteria for a diseased leaf
def is_diseased_contour(contour):
    # You can define your own criteria here based on contour properties
    # For example, you can check contour area, aspect ratio, etc.
    # Return True if the contour meets the criteria, False otherwise
    # Example criteria: return cv2.contourArea(contour) > threshold_area
    return False  # Change this to your actual criteria

# Loop through each class
for class_name in classes:
    class_input_directory = os.path.join(input_directory, class_name)
    class_output_directory = os.path.join(output_directory, class_name)

    # Create the output directory if it doesn't exist
    os.makedirs(class_output_directory, exist_ok=True)

    for filename in os.listdir(class_input_directory):
        if filename.endswith(".jpg"):  # Assuming your images are in jpg format
            image_path = os.path.join(class_input_directory, filename)
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply the Sobel operator for edge detection
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate the magnitude of gradients
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

            # Create a binary mask by thresholding the gradient magnitude
            threshold = 30  # Adjust the threshold as needed
            _, mask = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)

            # Convert the mask to the CV_8UC1 format
            mask = mask.astype(np.uint8)

            # Morphological dilation to highlight edges
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            # Morphological closing to further refine edges
            closing_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the closing_mask
            contours, _ = cv2.findContours(closing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a mask with the same size as the original image
            filled_roi_mask = np.zeros_like(mask)

            # Determine the contours that meet the criteria for diseased leaves
            diseased_contours = [contour for contour in contours if is_diseased_contour(contour)]

            # Draw the contours of diseased leaves on the filled_roi_mask
            cv2.drawContours(filled_roi_mask, diseased_contours, -1, (255), thickness=cv2.FILLED)

            # Use the filled_roi_mask to extract the diseased region from the original image
            extracted_diseased_region = cv2.bitwise_and(image, image, mask=filled_roi_mask)

            # Save the extracted diseased region
            result_path = os.path.join(class_output_directory, "result_" + filename)
            cv2.imwrite(result_path, extracted_diseased_region)

# return images of some leaf
# need to be polish, because some are not returning any leaf 