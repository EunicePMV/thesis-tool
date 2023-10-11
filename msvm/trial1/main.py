# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

import os
import cv2
import lxml.etree as ET

# Directory containing both images and XML annotation files
data_dir = './path_to_training_directory/'

# List all image files (you may need to filter by extension)
image_files = [file for file in os.listdir(data_dir) if file.endswith(('.jpg', '.jpeg', '.png'))]

# Iterate through image files and load annotations
for image_file in image_files:
    # Load the image
    image_path = os.path.join(data_dir, image_file)
    image = cv2.imread(image_path)

    # Load the corresponding XML annotation
    base_name, _ = os.path.splitext(image_file)
    xml_file = os.path.join(data_dir, f'{base_name}.xml')

    # Check if the XML file exists
    if os.path.exists(xml_file):
        # Parse the XML annotation
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Process the annotation data (e.g., extracting bounding boxes and labels)
        for object_elem in root.findall(".//object"):
            # Extract bounding box coordinates
            xmin = int(object_elem.find(".//xmin").text)
            ymin = int(object_elem.find(".//ymin").text)
            xmax = int(object_elem.find(".//xmax").text)
            ymax = int(object_elem.find(".//ymax").text)

            # Extract label
            label = object_elem.find(".//name").text

            # Draw bounding box on the image
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display or process the annotated image as needed
        cv2.imshow("Annotated Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"No XML annotation found for {image_file}")




# # Load your dataset and preprocess it
# X, y = load_and_preprocess_data()

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the multiclass SVM with RBF kernel
# svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')  # You can adjust C and gamma
# svm_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = svm_classifier.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
