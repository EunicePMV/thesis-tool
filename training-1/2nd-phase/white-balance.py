import os
import cv2
import numpy as np

input_directory = "../balanced-dataset"
output_directory = "./dataset-balanced"

def gray_world(img):
    # Calculate the average color of the image
    avg_color = np.mean(img, axis=(0, 1))

    # Calculate the scaling factor for each channel
    scale_factors = [128 / val for val in avg_color]

    # Apply the scaling factors to balance the whiteness
    balanced_img = img * scale_factors

    # Clip the values to ensure they are in the valid range [0, 255]
    balanced_img = np.clip(balanced_img, 0, 255).astype(np.uint8)

    return balanced_img

for class_name in os.listdir(input_directory):
    class_directory = os.path.join(input_directory, class_name)

    # Create a corresponding output directory if it doesn't exist
    output_class_directory = os.path.join(output_directory, class_name)
    os.makedirs(output_class_directory, exist_ok=True)

    for filename in os.listdir(class_directory):
        image_path = os.path.join(class_directory, filename)

        # Read the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (300, 300))

        if img is not None:
            # Apply the Gray World algorithm
            balanced_img = gray_world(img)

            # Save the processed image to the output directory
            output_path = os.path.join(output_class_directory, filename)
            cv2.imwrite(output_path, balanced_img)
