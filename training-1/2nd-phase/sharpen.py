import cv2
import os
import numpy as np

def sharpen_images(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define the sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])

    # Loop through the images in the input directory
    for class_name in os.listdir(input_directory):
        class_directory = os.path.join(input_directory, class_name)
        output_class_directory = os.path.join(output_directory, class_name)

        # Create the output class directory if it doesn't exist
        if not os.path.exists(output_class_directory):
            os.makedirs(output_class_directory)

        for filename in os.listdir(class_directory):
            image_path = os.path.join(class_directory, filename)
            output_image_path = os.path.join(output_class_directory, filename)

            # Read the image
            image = cv2.imread(image_path)

            # Apply sharpening filter
            sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

            # Save the sharpened image
            cv2.imwrite(output_image_path, sharpened_image)

    print("Sharpening complete.")

# Example usage:
input_directory = './dataset-denoised/'
output_directory = './dataset-sharpened/'
sharpen_images(input_directory, output_directory)
