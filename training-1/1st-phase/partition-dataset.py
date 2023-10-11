import os
import random
import shutil

# Set the path to your main folder
main_folder = './dataset/rice-tungro/'

# Define the ratios for training, validation, and testing sets
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create directories for training, validation, and testing sets
train_dir = os.path.join(main_folder, 'train')
val_dir = os.path.join(main_folder, 'validation')
test_dir = os.path.join(main_folder, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all the image files in the 'images' folder
image_files = [f for f in os.listdir(os.path.join(main_folder)) if f.endswith('.jpg')]

# Shuffle the image files randomly
random.shuffle(image_files)

# Calculate the number of images for each set
total_images = len(image_files)
num_train = int(total_images * train_ratio)
num_val = int(total_images * val_ratio)

# Copy images to the appropriate directories
for i, image_file in enumerate(image_files):
    source_path = os.path.join(main_folder, image_file)
    
    if i < num_train:
        dest_path = os.path.join(train_dir, image_file)
    elif i < num_train + num_val:
        dest_path = os.path.join(val_dir, image_file)
    else:
        dest_path = os.path.join(test_dir, image_file)
    
    shutil.copy(source_path, dest_path)

print(f"Total images: {total_images}")
print(f"Training set: {num_train} images")
print(f"Validation set: {num_val} images")
print(f"Testing set: {total_images - num_train - num_val} images")
