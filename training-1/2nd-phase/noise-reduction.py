# import cv2
# import os

# # Define the main dataset folder
# main_dataset_folder = '../balanced-dataset/'

# # Define a function to apply denoising to a directory of images
# def denoise_images_in_directory(input_dir, output_dir):
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.jpg'):  # Adjust the file extension as needed
#             image_path = os.path.join(input_dir, filename)
#             img = cv2.imread(image_path)
            
#             # Apply your chosen denoising method here
#             # For example, we'll use Gaussian blur in this case
#             img_denoised = cv2.GaussianBlur(img, (5, 5), 0)  # Adjust the kernel size as needed
            
#             # Save the denoised image to the output directory
#             output_path = os.path.join(output_dir, filename)
#             cv2.imwrite(output_path, img_denoised)

# # Loop through the subfolders (each class)
# for class_folder in os.listdir(main_dataset_folder):
#     class_folder_path = os.path.join(main_dataset_folder, class_folder)
    
#     # Check if it's a directory (not a file)
#     if os.path.isdir(class_folder_path):
#         # Create an output directory for denoised images for this class
#         output_class_folder = os.path.join(main_dataset_folder, f'denoised_{class_folder}')
#         os.makedirs(output_class_folder, exist_ok=True)
        
#         # Apply denoising to images in this class and save them in the output directory
#         denoise_images_in_directory(class_folder_path, output_class_folder)


# import cv2
# import os

# def denoise_images_in_dataset(main_dataset_folder, denoised_output_folder, denoising_method='Gaussian', kernel_size=(5, 5)):
#     """
#     Apply noise assessment and denoising to images in a dataset and save denoised images per class.

#     Args:
#         main_dataset_folder (str): Path to the main dataset folder containing subfolders for each class.
#         denoised_output_folder (str): Path to the folder where denoised images will be saved.
#         denoising_method (str, optional): Denoising method to use (e.g., 'Gaussian', 'Median', 'Bilateral').
#         kernel_size (tuple, optional): Kernel size for denoising (e.g., (5, 5)).
#     """
#     # Loop through the subfolders (each class)
#     for class_folder in os.listdir(main_dataset_folder):
#         class_folder_path = os.path.join(main_dataset_folder, class_folder)
        
#         # Check if it's a directory (not a file)
#         if os.path.isdir(class_folder_path):
#             # Create an output directory for denoised images for this class
#             output_class_folder = os.path.join(denoised_output_folder, f'denoised_{class_folder}')
#             os.makedirs(output_class_folder, exist_ok=True)
            
#             # Define a function to apply denoising to a directory of images
#             def denoise_images_in_directory(input_dir, output_dir):
#                 for filename in os.listdir(input_dir):
#                     if filename.endswith('.jpg'):  # Adjust the file extension as needed
#                         image_path = os.path.join(input_dir, filename)
#                         img = cv2.imread(image_path)
                        
#                         # Apply the chosen denoising method
#                         if denoising_method == 'Gaussian':
#                             img_denoised = cv2.GaussianBlur(img, kernel_size, 0)
#                         elif denoising_method == 'Median':
#                             img_denoised = cv2.medianBlur(img, kernel_size[0])
#                         elif denoising_method == 'Bilateral':
#                             img_denoised = cv2.bilateralFilter(img, kernel_size[0], sigma_color=75, sigma_space=75)
#                         else:
#                             raise ValueError("Invalid denoising method. Use 'Gaussian', 'Median', or 'Bilateral'.")
                        
#                         # Save the denoised image to the output directory
#                         output_path = os.path.join(output_dir, filename)
#                         cv2.imwrite(output_path, img_denoised)
            
#             # Apply denoising to images in this class and save them in the output directory
#             denoise_images_in_directory(class_folder_path, output_class_folder)

# # Example usage:
# main_dataset_folder = '../balanced-dataset'
# denoised_output_folder = 'dataset-denoised'
# denoising_method = 'Gaussian'
# kernel_size = (5, 5)

# denoise_images_in_dataset(main_dataset_folder, denoised_output_folder, denoising_method, kernel_size)



import cv2
import os

def denoise_images_in_class(class_folder, main_dataset_folder, output_folder, kernel_size=(5, 5)):
    """
    Denoise images in a class folder and save denoised images to the output folder.

    Args:
        class_folder (str): The name of the class folder containing images to denoise.
        main_dataset_folder (str): The path to the main dataset folder.
        output_folder (str): The path to the folder where denoised images will be saved.
        kernel_size (tuple): The size of the Gaussian blur kernel. Default is (5, 5).
    """
    class_path = os.path.join(main_dataset_folder, class_folder)
    for filename in os.listdir(class_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(class_path, filename)
            img = cv2.imread(image_path)
            
            # Split the image into RGB channels
            r_channel, g_channel, b_channel = cv2.split(img)
            
            # Apply denoising (Gaussian blur) to each channel
            r_denoised = cv2.GaussianBlur(r_channel, kernel_size, 0)
            g_denoised = cv2.GaussianBlur(g_channel, kernel_size, 0)
            b_denoised = cv2.GaussianBlur(b_channel, kernel_size, 0)
            
            # Merge the denoised channels back into an RGB image
            img_denoised = cv2.merge((r_denoised, g_denoised, b_denoised))
            
            # Save the denoised image to the output folder
            output_image_path = os.path.join(output_folder, class_folder, filename)
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, img_denoised)

def denoise_images_in_dataset(main_dataset_folder, output_folder, kernel_size=(5, 5)):
    """
    Denoise images in all class folders within the main dataset folder and save denoised images to the output folder.

    Args:
        main_dataset_folder (str): The path to the main dataset folder.
        output_folder (str): The path to the folder where denoised images will be saved.
        kernel_size (tuple): The size of the Gaussian blur kernel. Default is (5, 5).
    """
    # Loop through subfolders (classes) in the main dataset folder
    for class_folder in os.listdir(main_dataset_folder):
        if os.path.isdir(os.path.join(main_dataset_folder, class_folder)):
            denoise_images_in_class(class_folder, main_dataset_folder, output_folder, kernel_size)

# Example usage:
main_dataset_folder = './dataset-balanced'
output_folder = './dataset-denoised'
denoise_images_in_dataset(main_dataset_folder, output_folder)
