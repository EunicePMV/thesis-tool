# Canny edge detection

import os
import cv2

# Function to apply Canny edge detection and save images in the original format
def apply_canny_edge_detection(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each class directory in the input directory
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)

        # Ensure the output directory for the current class exists
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Loop through images in the current class directory
        for image_file in os.listdir(class_dir):
            if image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(class_dir, image_file)

                # Read the image
                image = cv2.imread(image_path)

                # Apply Canny edge detection
                edges = cv2.Canny(image, 100, 200)  # You can adjust these threshold values

                # Save the result in the class-specific output directory with the original format
                output_path = os.path.join(class_output_dir, image_file)
                cv2.imwrite(output_path, edges, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # Maintain original quality for JPEG images

# Specify your input and output directories here
input_directory = "./dataset-bilateralFil"  # Replace with the path to your main directory
output_directory = "./dataset-edge_detected"  # Replace with the path where you want to save the results

apply_canny_edge_detection(input_directory, output_directory)