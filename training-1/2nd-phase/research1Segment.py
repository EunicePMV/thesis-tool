import os
import cv2
import numpy as np

# Define the main directory containing the class subdirectories
main_directory = './dataset-balanced'

# Define the output directory
output_directory = './dataset-bgRemove'

# Define a function to apply background markers to an image
def apply_background_marker(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Step 1: Non-green background marker
    B, G, R = cv2.split(image)
    ExG = 2 * G - R - B
    ExR = 1.4 * R - G - B
    diff = ExG - ExR
    threshold = np.mean(diff) - np.std(diff)
    non_green_mask = (diff <= threshold) | (R > G)
    non_green_mask = (R > 200) & (G > 220) & (B > 200) & (R < 256) & (G < 256) & (B < 256) | (R < 30) & (G < 30) & (B < 30)

    # Step 2: Texture background marker
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy_threshold = 220
    entropy = cv2.filter2D(gray_image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    texture_mask = entropy > entropy_threshold

    # Combine the masks
    combined_mask = non_green_mask | texture_mask

    # Step 3: Marker cleanup
    min_area_threshold = 100
    distance_threshold = 50
    marker_mask = np.zeros_like(combined_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > min_area_threshold:
            distance = cv2.pointPolygonTest(contour, (image.shape[1] // 2, image.shape[0] // 2), True)
            if distance > distance_threshold:
                cv2.drawContours(marker_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Save the marker mask
    marker_output_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(marker_output_path, marker_mask)

# Loop through the class subdirectories and process images
for class_directory in os.listdir(main_directory):
    class_path = os.path.join(main_directory, class_directory)
    if os.path.isdir(class_path):
        for image_file in os.listdir(class_path):
            if image_file.endswith(".jpg") or image_file.endswith(".png"):
                image_path = os.path.join(class_path, image_file)
                apply_background_marker(image_path)


# STOP HERE:
# 1. try method from research - found git repo Nantheera Anantrasirichai*, Sion Hannuna and Nishan Canagarajah
# 2. search for more detailed dataset
# if research is not done successfully go back with extractEdges.py

# OPTION:
# 1. try the research 1 - not implemented well
# 2. try the research 2 - 
# 3. continue my edge segmentation

# ACCORDING TO THE RESEARCH: CURRENT TRIAL
# 1. Preprocessing
#   - resize & white balance
# 2. Main processing:
#   - background marker
#   - leaf marker
#   - initial leaf
#   - leaf shape refinement