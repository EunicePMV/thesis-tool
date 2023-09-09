# ../images/K-blb_3.jpg

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the RGB image
rgb_image = cv2.imread('../images/blb_1.jpg')  # Replace 'your_rgb_image.jpg' with your image file path
rgb_image = cv2.resize(rgb_image, (300, 450))

# Convert the BGR image to RGB
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

# Convert the RGB image to float32
rgb_image = rgb_image.astype(np.float32) / 255.0

# Split the RGB image into R, G, and B channels
R, G, B = cv2.split(rgb_image)

# Calculate the intensity (I) of the image
I = (R + G + B) / 3.0

# Calculate the minimum (minRGB) and maximum (maxRGB) of the RGB channels
minRGB = np.minimum(np.minimum(R, G), B)
maxRGB = np.maximum(np.maximum(R, G), B)

# Calculate the saturation (S) of the image
S = 1 - (minRGB / I)

# Initialize the hue (H) channel
H = np.zeros_like(I)

# Calculate the hue (H) channel for each pixel
for i in range(len(rgb_image)):
    for j in range(len(rgb_image[0])):
        num = 0.5 * ((R[i][j] - G[i][j]) + (R[i][j] - B[i][j]))
        den = ((R[i][j] - G[i][j])**2 + (R[i][j] - B[i][j]) * (G[i][j] - B[i][j]))**0.5
        theta = np.arccos(np.clip(num / (den + 1e-10), -1.0, 1.0))
        H[i][j] = theta if B[i][j] <= G[i][j] else (2.0 * np.pi - theta)

# Normalize H to the range [0, 1]
H /= (2.0 * np.pi)

# Stack H, S, and I channels to create the HSI image
hsi_image = np.dstack((H, S, I))

cv2.imwrite('./images/hsi_image.jpg', hsi_image)

# Display the HSI image
cv2.imshow('HSI Image', hsi_image)
cv2.waitKey(0)
cv2.destroyAllWindows()