import cv2
import numpy as np
import os

# Define functions for background marker techniques
def compute_exg_exr_threshold(image):
    R, G, B = image[:, :, 2], image[:, :, 1], image[:, :, 0]
    ExG = 2 * G - R - B
    ExR = 1.4 * R - G - B
    return ExG, ExR

def apply_exg_exr_threshold(ExG, ExR):
    threshold = np.mean(ExG) - np.std(ExG)
    mask = ((ExG - ExR) > threshold) | (R > G)
    return mask

def apply_texture_background(image, threshold=220):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -threshold
    )
    return entropy_mask

def clean_up_markers(markers, min_area=100, distance_threshold=50):
    # Find contours of markers
    contours, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask to remove small markers
    cleaned_markers = np.zeros_like(markers)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, _, _ = cv2.boundingRect(contour)
            if cv2.pointPolygonTest(contour, (x, y), True) > distance_threshold:
                cv2.drawContours(cleaned_markers, [contour], -1, 1, thickness=cv2.FILLED)
    
    return cleaned_markers

# Define the directory containing your images
data_dir = "./dataset-balanced"
output_dir = "./dataset-bgMark"

# Loop through each image in the directory
for filename in os.listdir(data_dir):
    if filename.endswith(".jpg"):  # Adjust the file format as needed
        image = cv2.imread(os.path.join(data_dir, filename))
        
        # Apply background marker techniques
        ExG, ExR = compute_exg_exr_threshold(image)
        exg_exr_mask = apply_exg_exr_threshold(ExG, ExR)
        texture_mask = apply_texture_background(image)
        
        # Combine the background markers
        background_markers = exg_exr_mask | texture_mask
        
        # Clean up the markers
        cleaned_markers = clean_up_markers(background_markers)
        
        # Create markers for leaf region (you can use other techniques here)
        leaf_markers = np.zeros_like(cleaned_markers)
        
        # Perform segmentation using watershed algorithm
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, markers = cv2.connectedComponents(leaf_markers)
        markers = markers + 1
        markers[cleaned_markers == 1] = 0
        
        # Apply watershed algorithm
        markers = cv2.watershed(image, markers)
        segmented_image = np.zeros_like(image)
        segmented_image[markers == -1] = [0, 0, 255]  # Mark watershed boundaries in red
        
        # Extract and save the segmented leaf regions
        segmented_leafs = np.zeros_like(image)
        segmented_leafs[markers > 1] = [0, 255, 0]  # Mark leaf regions in green
        
        cv2.imwrite(os.path.join(output_dir, filename), segmented_leafs)

# STOP HERE:
# 1. try method from research - found git repo Nantheera Anantrasirichai*, Sion Hannuna and Nishan Canagarajah
# 2. search for more detailed dataset
# if research is not done successfully go back with extractEdges.py

# OPTION:
# 1. try the research 1 - not implemented well
# 2. try the research 2 - 
# 3. continue my edge segmentation

# ACCORDING TO THE RESEARCH: CURRENT TRIAL -> HERE - try to fix
# 1. Preprocessing
#   - resize & white balance
# 2. Main processing:
#   - background marker
#   - leaf marker
#   - initial leaf
#   - leaf shape refinement