import cv2
import os

def apply_bilateral_filter(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Loop through each image in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust the file extensions as needed
            # Read the image
            image_path = os.path.join(input_directory, filename)
            img = cv2.imread(image_path)

            # Apply bilateral filter
            filtered_img = cv2.bilateralFilter(img, d=10, sigmaColor=75, sigmaSpace=75)  # You can adjust these parameters

            # Save the filtered image to the output directory
            output_path = os.path.join(output_directory, filename)
            cv2.imwrite(output_path, filtered_img)

main_directory = "./dataset-balanced"
output_base_directory = "./dataset-bilateralFil"

classes = ["bacterial-leaf-blight", "rice-blast", "rice-tungro", "healthy"]

for class_name in classes:
    input_directory = os.path.join(main_directory, class_name)
    output_directory = os.path.join(output_base_directory, class_name)
    apply_bilateral_filter(input_directory, output_directory)
