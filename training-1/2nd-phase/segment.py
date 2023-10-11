# # GPT Steps:
# 1. thresholding (otsu)
# 2. morphological operation
# 3. contour detection
# 4. contour filtering
# 5. ROI extraction
# import cv2
# import os

# def otsu_threshold(input_dir, output_dir):
#     # Create the output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Loop through each class directory
#     for class_name in os.listdir(input_dir):
#         class_path = os.path.join(input_dir, class_name)
        
#         # Create a subdirectory in the output directory for the class
#         class_output_dir = os.path.join(output_dir, class_name)
#         if not os.path.exists(class_output_dir):
#             os.makedirs(class_output_dir)

#         # Loop through each image in the class directory
#         for image_name in os.listdir(class_path):
#             image_path = os.path.join(class_path, image_name)

#             # Read the image
#             image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#             # Apply Otsu thresholding
#             _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#             # Save the thresholded image to the class output directory
#             output_path = os.path.join(class_output_dir, image_name)
#             cv2.imwrite(output_path, thresholded)

# input_directory = "./dataset-bilateralFil"
# output_directory = "./dataset-threshold"

# otsu_threshold(input_directory, output_directory)


import cv2
import os
import numpy as np

def otsu_threshold(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each class directory
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        
        # Create a subdirectory in the output directory for the class
        class_output_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        # Loop through each image in the class directory
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # Read the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Apply Otsu thresholding
            _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply morphological operation (Erosion in this case)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # You can adjust the kernel size
            eroded = cv2.erode(thresholded, kernel, iterations=1)

            # Save the result of the morphological operation to the class output directory
            output_path = os.path.join(class_output_dir, image_name)
            cv2.imwrite(output_path, eroded)

input_directory = "./dataset-bilateralFil"
output_directory = "./dataset-threshold_morph"

otsu_threshold(input_directory, output_directory)
