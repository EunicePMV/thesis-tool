import os
import random
import shutil

# Directory containing both images and XML annotation files
data_dir = '../images/bacterial-leaf-blight/'

# Directory where the training and test sets will be saved
train_dir = 'path_to_training_directory'
test_dir = 'path_to_test_directory'

# Create the training and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get a list of all files (both images and XML)
all_files = os.listdir(data_dir)

# Randomly shuffle the list of files
random.shuffle(all_files)

# Calculate the number of files for training and testing
total_files = len(all_files)
train_ratio = 0.8  # You can adjust the ratio as needed
num_train = int(total_files * train_ratio)
num_test = total_files - num_train

# Split the files into training and test sets
train_files = all_files[:num_train]
test_files = all_files[num_train:]

# Copy the selected files to their respective directories
for filename in train_files:
    base_name, file_extension = os.path.splitext(filename)
    if file_extension.lower() in ('.jpg', '.jpeg', '.png'):
        image_path = os.path.join(data_dir, filename)
        xml_path = os.path.join(data_dir, f'{base_name}.xml')

        # Copy the image and XML to the training directory
        shutil.copy(image_path, os.path.join(train_dir, filename))
        shutil.copy(xml_path, os.path.join(train_dir, f'{base_name}.xml'))

for filename in test_files:
    base_name, file_extension = os.path.splitext(filename)
    if file_extension.lower() in ('.jpg', '.jpeg', '.png'):
        image_path = os.path.join(data_dir, filename)
        xml_path = os.path.join(data_dir, f'{base_name}.xml')

        # Copy the image and XML to the test directory
        shutil.copy(image_path, os.path.join(test_dir, filename))
        shutil.copy(xml_path, os.path.join(test_dir, f'{base_name}.xml'))
