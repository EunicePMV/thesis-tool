import cv2
import os

# Define the main directory where your class folders are located
main_directory = "./dataset-threshold_morph"

# Define the list of class names
class_names = ["bacterial-leaf-blight", "rice-tungro", "rice-blast", "healthy"]

# Function to apply contour detection to images in a class directory
def apply_contour_detection(class_dir):
    for filename in os.listdir(class_dir):
        if filename.endswith(".jpg"):  # You can change the image file extension as needed
            image_path = os.path.join(class_dir, filename)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply contour detection
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on the original image
            img_with_contours = img.copy()
            cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)  # Green color for contours
            
            # Display or save the image with contours
            cv2.imshow("Image with Contours", img_with_contours)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Loop through each class and apply contour detection
for class_name in class_names:
    class_dir = os.path.join(main_directory, class_name)
    apply_contour_detection(class_dir)
