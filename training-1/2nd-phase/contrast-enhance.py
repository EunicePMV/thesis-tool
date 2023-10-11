import cv2
import os
import numpy as np

def enhance_contrast(image):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    
    # Merge the enhanced L channel with the A and B channels
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    
    # Convert the LAB image back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

input_directory = "./dataset-balanced"
output_directory = "./dataset-contrast_enhanced"

# List of class directories
class_names = ["bacterial-leaf-blight", "rice-tungro", "rice-blast", "healthy"]

for class_name in class_names:
    # Create the output directory for this class
    class_output_dir = os.path.join(output_directory, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    # Directory containing images for this class
    class_input_dir = os.path.join(input_directory, class_name)

    # Iterate through images in the class directory
    for filename in os.listdir(class_input_dir):
        if filename.endswith(".jpg"):  # Adjust the file extension if necessary
            # Load the image
            image_path = os.path.join(class_input_dir, filename)
            image = cv2.imread(image_path)

            # Apply contrast enhancement
            enhanced_image = enhance_contrast(image)

            # Save the enhanced image to the class output directory
            output_path = os.path.join(class_output_dir, filename)
            cv2.imwrite(output_path, enhanced_image)
