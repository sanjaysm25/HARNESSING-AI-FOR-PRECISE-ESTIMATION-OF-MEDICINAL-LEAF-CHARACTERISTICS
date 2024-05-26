import os
import cv2
import numpy as np

# Directory containing your dataset
dataset_dir = r'C:\Users\hp\Desktop\mAJOR PROJECT\Aryvedic\major project codes\major project final codes\dataset2'


# Desired size for resizing
desired_size = (224, 224)

# Function to resize and normalize images
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Resize the image
    resized_img = cv2.resize(img, desired_size)
    
    # Normalize pixel values
    normalized_img = resized_img / 255.0
    
    return normalized_img

# Iterate through each image in the dataset directory
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        # Get the full path to the image
        image_path = os.path.join(root, file)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        
        # Save or overwrite the preprocessed image
        cv2.imwrite(image_path, (preprocessed_image * 255).astype(np.uint8))
