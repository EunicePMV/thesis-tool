import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your image dataset, replace this with your dataset loading code.
# Your dataset should be a 2D NumPy array where each row is a feature vector.
dataset = ...

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to your data and transform it
normalized_dataset = scaler.fit_transform(dataset)

# Generate random data for demonstration purposes
np.random.seed(42)
dataset = np.random.rand(100, 10)  # 100 samples with 10 features

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to your data and transform it
normalized_dataset = scaler.fit_transform(dataset)

# Check the mean and standard deviation of the normalized data
print("Mean:", np.mean(normalized_dataset, axis=0))
print("Standard Deviation:", np.std(normalized_dataset, axis=0))
