import os
import cv2

# Function to enhance an image
def enhance_image(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_a_channel = clahe.apply(a_channel)
    enhanced_b_channel = clahe.apply(b_channel)
    enhanced_lab_image = cv2.merge((l_channel, enhanced_a_channel, enhanced_b_channel))
    enhanced_rgb_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    return enhanced_rgb_image

# Function to process images in class folders
def process_class_folders(main_directory, output_directory):
    class_folders = os.listdir(main_directory)

    for class_folder in class_folders:
        class_path = os.path.join(main_directory, class_folder)
        output_class_directory = os.path.join(output_directory, class_folder)
        os.makedirs(output_class_directory, exist_ok=True)

        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            # Enhance the image
            enhanced_image = enhance_image(image)

            # Save the enhanced image in the output class directory
            output_image_path = os.path.join(output_class_directory, filename)
            cv2.imwrite(output_image_path, enhanced_image)

    print("Color enhancement completed.")

# Example usage:
main_directory = './dataset-sharpened'
output_directory = './dataset-color_enhanced'

process_class_folders(main_directory, output_directory)
