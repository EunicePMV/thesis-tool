# STANDARD PSO:
# velocity and particle position is initialized randomly
# no kfold

import os
import numpy as np
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage import io, color
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from joblib import dump, Parallel, delayed
import cv2

# Define the main dataset directory
dataset_dir = '../trial-dataset2-segmented'

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
    return lbp_hist

# Function to compute Histogram of Oriented Gradients (HOG) for an image
def compute_hog(image):
    # Convert the image to grayscale
    gray = color.rgb2gray(image)

    # Compute HOG features
    hog_features, _ = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)

    return hog_features

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
    # return mean, std
    return stat_features

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
        
        # Compute LBP histogram for the image
        lbp_histogram = compute_lbp_histogram(image)
        
        # Compute HOG features for the image
        hog_features = compute_hog(image)

        # Compute revise GLCM features
        glcm_features = compute_glcm(image)

        # Compute the mean and standard deviation for Color features
        color_features = compute_color_features(image)

        # Compute revise Color Histogram features
        colorHistogram_features = compute_color_histogram(image)

        # Compute CCV rev2
        ccv_features = compute_ccv(image)

        # Compute stat features
        stat_features = compute_stat_features(image)

        # Concatenate LBP and color histograms to create the feature vector
        combined_features = np.concatenate((lbp_histogram, glcm_features, color_features,
                                            colorHistogram_features, ccv_features, hog_features, 
                                            stat_features))

        # Append the feature vector and label to the lists
        features.append(combined_features)
        labels.append(class_label)

# Convert the lists to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X = scaler.fit_transform(X)

def evaluate_particle(X_selected, y, particle, threshold):
    # Apply threshold θ to select features
    selected_features = np.where(particle > threshold)[0]

    # Calculate the objective function value
    if np.sum(selected_features) == 0:
        score = 1.0  # Avoid division by zero
    else:
        X_selected_particle = X_selected[:, selected_features]
        classifier = SVC(kernel='rbf', C=10, verbose=10, probability=True)
        classifier.fit(X_selected_particle, y)
        y_pred = classifier.predict(X_selected_particle)
        accuracy = accuracy_score(y, y_pred)
        score = 1.0 - accuracy  # Minimize 1 - accuracy

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

# Perform feature selection with custom PSO
selected_features = custom_pso_parallel(X, y, num_particles=30, num_iterations=50, c1=1.49618, c2=1.49618, w=0.7298, threshold=0.6)

# Save the indices of the selected features
selected_feature_indices = np.where(selected_features)[0]
np.save('./features.npy', selected_feature_indices)

# Apply selected features
X_selected = X[:, np.where(selected_features)[0]]

# Split the data into training and testing sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train an SVM classifier with RBF kernel
classifier = SVC(kernel='rbf', C=10, verbose=10, probability=True)

# Fit the model on the training set
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = classifier.predict(X_test)

dump(classifier, './MSVM.joblib')
dump(scaler, './scaler.joblib')

# Get the number of features
num_features = X_selected.shape[1]
print("Number of Features:", num_features)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy on test set:", accuracy)

# Generate and print the classification report
class_report = classification_report(y_test, y_test_pred)
print("Classification Report on test set:\n", class_report)