import os
import cv2
import numpy as np

from skimage import io, color
from skimage.feature import local_binary_pattern, hog
from sklearn.preprocessing import StandardScaler
from joblib import load

from skimage.color import rgb2hsv
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

# Canny edge detection parameters
lower_threshold = 15
upper_threshold = 150

# Mask dilation and erosion parameters
dilation_kernel = np.ones((3, 3), np.uint8)
erosion_kernel = np.ones((3, 3), np.uint8)

# Morphological iterations
MIdilation = 2
MIerosion = 2

# Function to create a mask that fills the entire closed region within the outer contour
def create_convex_polygon_mask(edges):
    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) == 0:
        return np.zeros_like(edges)  # Return an empty mask if no contours are found

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask with the same size as the edges
    mask = np.zeros_like(edges)

    # Fill the largest contour
    cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

    return mask

# Function to perform Canny edge detection and mask dilation/erosion on images in a directory
def preprocess_segment(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Read the image
            image = cv2.imread(input_path)
            image = cv2.resize(image, (300, 300))

            # Apply Canny edge detection
            edges = cv2.Canny(image, lower_threshold, upper_threshold)

            # Dilation and erosion on edges
            edges = cv2.erode(cv2.dilate(edges, dilation_kernel), erosion_kernel)

            # Create a complete fill mask
            mask = create_convex_polygon_mask(edges)

            # Apply Gaussian blur to the mask
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            # Additional iterations of dilation and erosion
            mask = cv2.dilate(mask, None, iterations=MIdilation)
            mask = cv2.erode(mask, None, iterations=MIerosion)

            # Multiply the mask by 3
            mask_stack = mask * 3

            # Invert the result
            # mask_stack = cv2.bitwise_not(mask)

            print(filename)

            # Apply background subtraction
            foreground = cv2.bitwise_and(image, image, mask=mask_stack)

            # Save the result
            cv2.imwrite(output_path, foreground, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

# Function to compute LBP histogram for an image
def compute_lbp_histogram(image):
    # Convert the image to grayscale and to integer data type
    gray = color.rgb2gray(image)
    gray = (gray * 255).astype(np.uint8)

    # Compute LBP features
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    # lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist
    
# Function to compute Histogram of Oriented Gradients (HOG) for an image
def compute_hog(image):
    # Convert the image to grayscale
    gray = color.rgb2gray(image)

    # Compute HOG features
    hog_features, _ = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)

    return hog_features

# Function to compute Gray Level Co-Occurence Matrix for an image 
def compute_glcm(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Calculate the GLCM matrix
    distances = [1]  # Specify the distances between pixels to consider
    angles = [0]  # Specify the angles to consider
    glcm = graycomatrix(image, distances, angles, levels=256, symmetric=True, normed=True)
    
    # Calculate the GLCM features
    energy = graycoprops(glcm, 'energy').mean()
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    
    # Return the calculated features
    return energy, contrast, correlation, homogeneity

# Function to compute RGB, HSV and LAB feature for an image
def compute_color_features(img):
    # Convert the image to different color spaces
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split channels for RGB image
    r, g, b = cv2.split(img_rgb)

    # Calculate mean and standard deviation for RGB channels
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    std_r, std_g, std_b = np.std(r), np.std(g), np.std(b)

    # Split channels for HSV image
    h, s, v = cv2.split(img_hsv)

    # Calculate mean and standard deviation for HSV channels
    mean_h, mean_s, mean_v = np.mean(h), np.mean(s), np.mean(v)
    std_h, std_s, std_v = np.std(h), np.std(s), np.std(v)

    # Split channels for LAB image
    l, a, b_lab = cv2.split(img_lab)

    # Calculate mean and standard deviation for LAB channels
    mean_l, mean_a, mean_b_lab = np.mean(l), np.mean(a), np.mean(b_lab)
    std_l, std_a, std_b_lab = np.std(l), np.std(a), np.std(b_lab)

    return [mean_r, mean_g, mean_b, std_r, std_g, std_b, mean_h, mean_s, mean_v, std_h, std_s, std_v, mean_l, mean_a, mean_b_lab, std_l, std_a, std_b_lab]

# Function to compute color histogram (RGB, HSV & LAB) for an image
def compute_color_histogram(image):
    # Convert the image to RGB, HSV, and LAB color spaces
    rgb_image = image
    hsv_image = color.rgb2hsv(image)
    lab_image = color.rgb2lab(image)

    # Initialize histograms
    rgb_hist = []
    hsv_hist = []
    lab_hist = []

    # Compute histograms for each channel in each color space
    for i in range(3):  # Loop over channels (R, G, B)
        rgb_hist_channel, _ = np.histogram(rgb_image[:, :, i].ravel(), bins=256, range=(0, 256), density=True)
        hsv_hist_channel, _ = np.histogram(hsv_image[:, :, i].ravel(), bins=256, range=(0, 1), density=True)
        lab_hist_channel, _ = np.histogram(lab_image[:, :, i].ravel(), bins=256, range=(-128, 128), density=True)

        rgb_hist.extend(rgb_hist_channel)
        hsv_hist.extend(hsv_hist_channel)
        lab_hist.extend(lab_hist_channel)

    return np.concatenate((rgb_hist, hsv_hist, lab_hist))

# Function to compute Color Coherence Vector (CCV) for an image 
def compute_ccv(image, num_regions=4):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the HSV image into individual channels
    h, s, v = cv2.split(hsv_image)

    # Calculate the color histograms for each channel
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])

    # Normalize the histograms
    hist_h /= hist_h.sum()
    hist_s /= hist_s.sum()
    hist_v /= hist_v.sum()

    # Calculate the color coherence features
    color_coherence = np.concatenate((hist_h, hist_s, hist_v), axis=None)

    # Calculate the mean and standard deviation
    mean = np.mean(color_coherence)
    std_dev = np.std(color_coherence)

    return [mean, std_dev]

# Function to compute standard deviation & mean for an image
def compute_stat_features(image):
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    stat_features = np.concatenate((mean, std))

    return stat_features

input_dir = './input-dhan-shomadhan'
output_dir = './output-dhan-shomadhan'

os.makedirs(output_dir, exist_ok=True)

preprocess_segment(input_dir, output_dir)

fe_input_dir = output_dir
scaler = load('./export/base/5/scaler.joblib')
classifier = load('./export/base/5/MSVM.joblib') 

for filename in os.listdir(fe_input_dir):
    if filename.endswith(".jpg"):
        input_path = os.path.join(fe_input_dir, filename)
        user_image = io.imread(input_path)

        user_image = cv2.resize(user_image, (100, 100))

        lbp_histogram = compute_lbp_histogram(user_image)
        hog_features = compute_hog(user_image)
        revise_glcm = compute_glcm(user_image)
        revise_color = compute_color_features(user_image)
        revise_colorHistogram = compute_color_histogram(user_image)
        rev2_ccv = compute_ccv(user_image)
        stat_features = compute_stat_features(user_image)

        # combined_features = np.concatenate((revise_clbp, lbp_histogram, revise_glcm, revise_color,
        #                                     revise_colorHistogram, rev2_ccv, hog_features, stat_features))
        
        combined_features = np.concatenate((lbp_histogram, revise_glcm, revise_color,
                                            revise_colorHistogram, rev2_ccv, hog_features, stat_features))

        features = []
        features.append(combined_features)
        combined_features = np.array(features)
        X = scaler.transform(combined_features)

        # Predict and print results
        pred = classifier.predict(X)
        print(f"{filename}: predicted class {pred[0]}")