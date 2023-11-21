import os
import cv2
import numpy as np

from skimage import io, color
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from joblib import load
from skimage.color import rgb2hsv
from skimage import img_as_ubyte

# LBP
RADIUS=1
N_POINTS=8*RADIUS

# CCV
NUM_BINS=8

# COLOR HISTOGRAM
CHANNELS=(0,1,2)
NUM_BINS_CH=256
RANGES=(0,256)

def segment_leaf(image, output_path, LB=191, UB=251):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    v = cv2.equalizeHist(v)
    v = cv2.convertScaleAbs(v, alpha=1.25)
    _, v = cv2.threshold(v, LB, UB, cv2.THRESH_BINARY)
    
    # Thresholding based segmentation
    leaf = cv2.bitwise_or(s, v)
    mask = leaf
    
    # Thresholding
    _, mask = cv2.threshold(mask, LB, UB, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(image, image, mask=mask)

    cv2.imwrite(output_path, mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def compute_lbp_histogram(image, radius, n_points):
    # Convert the image to grayscale and to integer data type
    gray = color.rgb2gray(image)
    gray = (gray * 255).astype(np.uint8) # Comment out kapag nag eerror / warning

    # Compute LBP features
    lbp_image = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float") # Comment out kapag nag eerror / warning
    return lbp_hist

def compute_hog(image):
    # Convert the image to grayscale
    gray = color.rgb2gray(image)

    # Compute HOG features
    hog_features, _ = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)
    # hog_img = hog_img.flatten()

    return hog_features

def compute_ccv(image, num_bins):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Initialize the CCV histogram
    ccv_hist = np.zeros((num_bins, num_bins, num_bins))

    # Define the size of the regions
    region_width = lab_image.shape[1] // num_bins
    region_height = lab_image.shape[0] // num_bins

    # Calculate CCV for each region
    for i in range(num_bins):
        for j in range(num_bins):
            # Get the current region
            region = lab_image[j * region_height:(j + 1) * region_height,
                              i * region_width:(i + 1) * region_width]

            # Calculate the histogram for the region
            hist = cv2.calcHist([region], [0, 1, 2], None, [num_bins, num_bins, num_bins],
                                [0, 256, 0, 256, 0, 256])

            # Normalize the histogram and store it in the CCV
            hist /= np.sum(hist)
            ccv_hist += hist

    # Flatten the CCV histogram
    ccv_vector = ccv_hist.flatten()
        
    return ccv_vector

def compute_stat_features(image):
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    stat_features = np.concatenate((mean, std))
    # return mean, std
    return stat_features

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
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    
    # Return the calculated features
    return energy, contrast, correlation, homogeneity, dissimilarity

def compute_color_features(image):
    # Convert the image to different color spaces
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

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

def compute_color_histogram(image, channels, num_bins, ranges):
    hist_features = []

    for channel in channels:
        channel_hist = cv2.calcHist([image], [channel], None, [num_bins], ranges=ranges)
        channel_hist = channel_hist.flatten()
        hist_features.extend(channel_hist)
    
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    for channel in channels:
        channel_hist = cv2.calcHist([lab_image], [channel], None, [num_bins], ranges=ranges)
        channel_hist = channel_hist.flatten()
        hist_features.extend(channel_hist)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for channel in channels:
        channel_hist = cv2.calcHist([hsv_image], [channel], None, [num_bins], ranges=ranges)
        channel_hist = channel_hist.flatten()
        hist_features.extend(channel_hist)
        
    return hist_features

input_dir = '../finalized/testing' # main dataset directory 
output_dir = '../finalized/testing-fs' # main dataset directory 

# input_dir = '../google'
# output_dir = '../google-final'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".jpeg", ".png", ".JPG")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path)

        segment_leaf(img, output_path)

fe_input_dir = output_dir
scaler = load('./saved/fs/scaler.joblib')
classifier = load('./saved/fs/MSVM.joblib')
selected_features = np.load('./saved/fs/features.npy')

for filename in os.listdir(fe_input_dir):
    if filename.endswith((".jpg", ".jpeg", ".png", ".JPG")):
        input_path = os.path.join(fe_input_dir, filename)
        user_image = io.imread(input_path)

        user_image = cv2.resize(user_image, (100, 100))

        lbp_histogram = compute_lbp_histogram(user_image, RADIUS, N_POINTS)
        hog_feature = compute_hog(user_image)
        ccv_feature = compute_ccv(user_image, NUM_BINS)
        stat_feature = compute_stat_features(user_image)
        glcm_feature = compute_glcm(user_image)
        color_feature = compute_color_features(user_image)
        color_histogram = compute_color_histogram(user_image, CHANNELS, NUM_BINS_CH, RANGES)

        # Create the feature vector
        combined_features = np.concatenate((lbp_histogram, hog_feature, ccv_feature,
                                            stat_feature, glcm_feature, color_feature,
                                            color_histogram))

        features = []
        features.append(combined_features)
        combined_features = np.array(features)
        X = scaler.transform(combined_features)
        X_selected = X[:, selected_features]

        # Predict and print results
        pred = classifier.predict(X_selected)
        class_labels = classifier.classes_
        y_pred_prob = classifier.predict_proba(X_selected)
        for i, prob_vector in enumerate(y_pred_prob):
            class_probabilities = dict(zip(class_labels, prob_vector))

            # Convert probabilities to percentages and round to nearest tenths
            class_probabilities_percentage = {label: round(prob * 100, 1) for label, prob in class_probabilities.items()}

            print(f"{class_probabilities_percentage}")
        print(f"{filename}: predicted class {pred[0]}\n")