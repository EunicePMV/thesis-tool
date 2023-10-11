import os
import cv2
import numpy as np

def contrast_stretch(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    # Apply contrast stretch to the image
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return stretched

def process_images_in_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            output_path = os.path.join(output_class_dir, filename)

            # Load the image
            image = cv2.imread(image_path)

            # Apply contrast stretch
            stretched = contrast_stretch(image)

            # Save the processed image
            cv2.imwrite(output_path, stretched)

input_dir = "./dataset-balanced"
output_dir = "./dataset-contrast_stretched"

process_images_in_directory(input_dir, output_dir)
