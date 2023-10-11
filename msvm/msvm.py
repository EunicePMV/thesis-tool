import cv2
import numpy as np
import lxml.etree as ET

# Load the image
image_path = './images/blb-1.jpg'
image = cv2.imread(image_path)

# Load the corresponding XML annotation
xml_path = './images/blb-1.xml'
tree = ET.parse(xml_path)
root = tree.getroot()

# Loop through XML elements and extract annotation information
for annotation in root.findall(".//object"):
    # Extract bounding box coordinates
    xmin = int(annotation.find(".//xmin").text)
    ymin = int(annotation.find(".//ymin").text)
    xmax = int(annotation.find(".//xmax").text)
    ymax = int(annotation.find(".//ymax").text)

    # Extract label (class name)
    label = annotation.find(".//name").text

    # Draw bounding box on the image
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the annotated image
cv2.imshow("Annotated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
