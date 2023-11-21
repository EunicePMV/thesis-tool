import numpy as np
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage import color, io
import cv2

import os 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from joblib import dump, Parallel, delayed
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

def evaluate_particle(X_selected, y, particle, threshold):
    # Apply threshold θ to select features
    selected_features = np.where(particle > threshold)[0]

    # Calculate the objective function value
    if np.sum(selected_features) == 0:
        score = 1.0 # Avoid division by zero
    else:
        X_selected_particle = X_selected[:, selected_features]
        kfold = KFold(n_splits=4, shuffle=True)
        classifier = SVC(kernel='rbf', C=10, verbose=10, probability=True)
        
        # Perform cross-validation and generate a classification report for each fold
        for train_index, test_index in kfold.split(X_selected_particle):
            X_train_fold, X_test_fold = X_selected_particle[train_index], X_selected_particle[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            
            # Fit the scaler on the training data of this fold
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            
            # Use the fitted scaler to transform the test data of this fold
            X_test_fold = scaler.transform(X_test_fold)
            
            classifier.fit(X_train_fold, y_train_fold)
            y_pred = classifier.predict(X_test_fold)
            
            accuracy = accuracy_score(y_test_fold, y_pred)
            score = 1.0 - accuracy # Minimize 1 - accuracy

    return score, selected_features

def custom_pso_parallel(X, y, num_particles, num_iterations, c1, c2, w, threshold):
    num_features = X.shape[1]
    lb = [0] * num_features  # Lower bound for each feature (0: not selected, 1: selected)
    ub = [1] * num_features  # Upper bound for each feature

    # Initialize particle positions and velocities
    particles = np.random.uniform(0, 1, (num_particles, num_features))
    velocities = np.zeros((num_particles, num_features))
    personal_best_positions = particles.copy()
    personal_best_scores = np.ones(num_particles)

    # Find the global best particle
    global_best_index = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index]

    for iteration in range(num_iterations):
        # Evaluate particles in parallel
        results = Parallel(n_jobs=-1)(delayed(evaluate_particle)(X, y, particles[i], threshold) for i in range(num_particles))

        for i, (score, selected_features) in enumerate(results):
            # Update personal best position and score
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]

                # Update global best position
                if score < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = personal_best_positions[i]

        for i in range(num_particles):
            # Update particle velocities
            r1, r2 = np.random.rand(2)
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_positions[i] - particles[i]) + c2 * r2 * (
                        global_best_position - particles[i])

            # Update particle positions
            particles[i] = particles[i] + velocities[i]

            # Clamp particle positions to the lower and upper bounds
            particles[i] = np.clip(particles[i], lb, ub)

    return global_best_position

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

# Split the data into training and testing sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
dump(scaler, './saved/fs/scaler.joblib')

# Perform feature selection with custom PSO
selected_features = custom_pso_parallel(
                                X_train, 
                                y_train, 
                                num_particles=30, 
                                num_iterations=50, 
                                c1=1.49618, 
                                c2=1.49618, 
                                w=0.7298, 
                                threshold=0.6)

# Save the indices of the selected features
selected_feature_indices = np.where(selected_features)[0]
np.save('./saved/fs/features.npy', selected_feature_indices)

# Apply selected features 
X_selected = X_train[:, selected_feature_indices]

# Train multiclass SVM classifier with RBF kernel
classifier = SVC(kernel='rbf', C=10, verbose=10, probability=True)

kfold = KFold(n_splits=4, shuffle=True)

reports = []

# Perform cross-validation and generate a classification report for each fold
for train_index, test_index in kfold.split(X_selected):
    X_train_fold, X_test_fold = X_selected[train_index], X_selected[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    
    # Fit the scaler on the training data of this fold
    scaler_model = StandardScaler()
    X_train_fold = scaler_model.fit_transform(X_train_fold)
    
    # Use the fitted scaler to transform the test data of this fold
    X_test_fold = scaler_model.transform(X_test_fold)
    
    classifier.fit(X_train_fold, y_train_fold)
    y_pred = classifier.predict(X_test_fold)
    
    report = classification_report(y_test_fold, y_pred, output_dict=True)
    reports.append(report)

# Convert the list of reports to a dictionary with the fold number as the key
reports_dict = {f'fold-{i+1}': report for i, report in enumerate(reports)}

# Convert the dictionary to a JSON string
json_reports = json.dumps(reports_dict)

# Write the JSON string to a file
with open('./saved/fs/reports.json', 'w') as f:
    f.write(json_reports)

dump(classifier, './saved/fs/MSVM.joblib')

# Use the model to make predictions on the test set
X_test_selected = X_test[:, selected_feature_indices]
y_pred_test = classifier.predict(X_test_selected)

# Print the predicted classes
print(y_pred_test)
print(y_test)