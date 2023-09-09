import cv2
import numpy as np

# Load your segmented disease portion image
segmented_disease_image_path = './images/diseased_part.jpg'  # Replace with the path to your segmented image
segmented_disease_image = cv2.imread(segmented_disease_image_path)

# Extract RGB components and calculate mean and standard deviation
b, g, r = cv2.split(segmented_disease_image)
mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
std_b, std_g, std_r = np.std(b), np.std(g), np.std(r)

# Convert the segmented disease portion to HSV color space
hsv_segmented_disease = cv2.cvtColor(segmented_disease_image, cv2.COLOR_BGR2HSV)

# Extract HSV components and calculate mean
h, s, v = cv2.split(hsv_segmented_disease)
mean_h, mean_s, mean_v = np.mean(h), np.mean(s), np.mean(v)

# Convert the segmented disease portion to LAB color space
lab_segmented_disease = cv2.cvtColor(segmented_disease_image, cv2.COLOR_BGR2LAB)

# Extract LAB components and calculate mean
l, a, b_lab = cv2.split(lab_segmented_disease)
mean_l, mean_a, mean_b_lab = np.mean(l), np.mean(a), np.mean(b_lab)

# Print the results
print("RGB Mean (R, G, B):", mean_r, mean_g, mean_b)
print("RGB Std Dev (R, G, B):", std_r, std_g, std_b)
print("HSV Mean (H, S, V):", mean_h, mean_s, mean_v)
print("LAB Mean (L, A, B):", mean_l, mean_a, mean_b_lab)
