# no folds

import numpy as np
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage import color, io
import cv2

import os 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from joblib import dump
import json

# LBP
RADIUS=1
N_POINTS=8*RADIUS

# CCV
NUM_BINS=8

# COLOR HISTOGRAM
CHANNELS=(0,1,2)
NUM_BINS_CH=256
RANGES=(0,256)

dataset_dir = './dataset-segmented'

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

# Initialize lists to store features and labels
features = []
labels = []

# Loop through each subdirectory (class) in the dataset directory
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    class_label = class_name

    # Loop through the images in the class directory
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = io.imread(image_path)
        image = cv2.resize(image, (100, 100))
        
        lbp_histogram = compute_lbp_histogram(image, RADIUS, N_POINTS)
        hog_feature = compute_hog(image)
        ccv_feature = compute_ccv(image, NUM_BINS)
        stat_feature = compute_stat_features(image)
        glcm_feature = compute_glcm(image)
        color_feature = compute_color_features(image)
        color_histogram = compute_color_histogram(image, CHANNELS, NUM_BINS_CH, RANGES)

        # Create the feature vector
        combined_features = np.concatenate((lbp_histogram, hog_feature, ccv_feature,
                                            stat_feature, glcm_feature, color_feature,
                                            color_histogram))

        # Append the feature vector and label to the lists
        features.append(combined_features)
        labels.append(class_label)

# Convert the lists to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X = scaler.fit_transform(X)
dump(scaler, './saved/base/scaler.joblib')

# Split the data into training and testing sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a one-vs-one multiclass SVM classifier with RBF kernel
classifier = SVC(kernel='rbf', C=10, verbose=10, probability=True)

# Fit the model on the full training set
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = classifier.predict(X_test)
dump(classifier, './saved/base/MSVM.joblib')

# Get the number of features
num_features = X.shape[1]
print("Number of Features:", num_features)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy on test set:", accuracy)

# Generate and print the classification report
class_report = classification_report(y_test, y_test_pred, output_dict=True)
print("Classification Report on test set:\n", class_report)

# {
#     'blb': {'precision': 0.9850746268656716, 
#             'recall': 0.9705882352941176, 
#             'f1-score': 0.9777777777777777, 
#             'support': 68.0}, 
#     'hlt': {'precision': 0.925, 
#             'recall': 0.9866666666666667, 
#             'f1-score': 0.9548387096774195, 
#             'support': 75.0}, 
#     'rb': {'precision': 0.972972972972973, 
#            'recall': 0.972972972972973, 
#            'f1-score': 0.972972972972973, 
#            'support': 74.0}, 
#     'sb': {'precision': 1.0, 
#            'recall': 0.9436619718309859, 
#            'f1-score': 0.9710144927536231, 
#            'support': 71.0}, 
#     'accuracy': 0.96875, 
#     'macro avg': {'precision': 0.9707618999596612, 
#                   'recall': 0.9684724616911858, 
#                   'f1-score': 0.9691509882954483, 
#                   'support': 288.0}, 
#     'weighted avg': {'precision': 0.9700002591210612, 
#                      'recall': 0.96875, 
#                      'f1-score': 0.9689021565979257,
#                      'support': 288.0}
# }



# Save the classification report to a JSON file
# json_filename = 'classification_report.json'
# with open(json_filename, 'w') as json_file:
#     json.dump(class_report, json_file, indent=4)