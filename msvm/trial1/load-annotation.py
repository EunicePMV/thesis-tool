import cv2
import os
from bs4 import BeautifulSoup

# Define the directory containing your images and XML annotation files
dataset_dir = "./path_to_training_directory/"

# Function to parse XML annotation file and extract annotation coordinates
def parse_xml_annotation(xml_file_path):
    with open(xml_file_path, 'r') as file:
        xml_data = file.read()
    
    soup = BeautifulSoup(xml_data, 'xml')
    annotations = soup.find_all('object')

    bounding_boxes = []
    for annotation in annotations:
        xmin = int(annotation.find('xmin').text)
        ymin = int(annotation.find('ymin').text)
        xmax = int(annotation.find('xmax').text)
        ymax = int(annotation.find('ymax').text)
        bounding_boxes.append((xmin, ymin, xmax, ymax))
    
    return bounding_boxes

# Iterate through the images and annotations in the dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(dataset_dir, filename)
        xml_path = os.path.join(dataset_dir, filename.split('.')[0] + '.xml')
        
        if os.path.exists(xml_path):
            image = cv2.imread(image_path)
            annotations = parse_xml_annotation(xml_path)

            # Draw bounding boxes on the image
            for xmin, ymin, xmax, ymax in annotations:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Display the annotated image
            cv2.imshow('Annotated Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
