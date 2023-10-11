
import cv2
import os
import numpy as np

# Define the directory where your input images are stored
input_directory = "./dataset-bilateralFil"

# Define the directory where you want to save the output processed images
output_directory = './dataset-sobelOpt_morphDilationClosing'

# Create a list of class names
classes = ['bacterial-leaf-blight', 'rice-blast', 'rice-tungro', 'healthy']

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

            # Aspply the Sobel operator for edge detection
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate the magnitude of gradients
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

            # Apply dilation operation
            kernel = np.ones((5, 5), np.uint8)
            dilated_image = cv2.dilate(gradient_magnitude, kernel, iterations=1)

            # Apply closing operation
            closing_kernel = np.ones((10, 10), np.uint8)
            closing_image = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, closing_kernel)

            # Save the processed image
            processed_image_path = os.path.join(class_output_directory, "processed_" + filename)
            cv2.imwrite(processed_image_path, closing_image)

# './dataset-bilateralFil'
# './edge-extraction/12'

# STOP HERE: contour filtering conditions