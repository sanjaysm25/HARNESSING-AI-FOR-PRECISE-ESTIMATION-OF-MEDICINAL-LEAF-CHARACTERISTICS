import cv2
import csv
import os
import numpy as np
from skimage import feature

def extract_features(image_path, label, filename):
    # Read the leaf image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Texture features
    glcm = feature.graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = np.mean(feature.graycoprops(glcm, 'contrast'))
    correlation = np.mean(feature.graycoprops(glcm, 'correlation'))
    energy = np.mean(feature.graycoprops(glcm, 'energy'))
    homogeneity = np.mean(feature.graycoprops(glcm, 'homogeneity'))

    # Shape features
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        area = cv2.contourArea(contours[0])
        perimeter = cv2.arcLength(contours[0], True)
        
        # Avoid division by zero
        if cv2.boundingRect(contours[0])[0] != 0:
            aspect_ratio = cv2.boundingRect(contours[0])[1] / cv2.boundingRect(contours[0])[0]
        else:
            aspect_ratio = 0
    else:
        area = 0
        perimeter = 0
        aspect_ratio = 0

    # Color features
    mean_red = np.mean(image[:, :, 0])
    mean_green = np.mean(image[:, :, 1])
    mean_blue = np.mean(image[:, :, 2])
    median_red = np.median(image[:, :, 0])
    median_green = np.median(image[:, :, 1])
    median_blue = np.median(image[:, :, 2])
    std_red = np.std(image[:, :, 0])
    std_green = np.std(image[:, :, 1])
    std_blue = np.std(image[:, :, 2])

    # Combine all features into a list
    features = [label, filename, contrast, correlation, energy, homogeneity, area, perimeter, aspect_ratio,
                mean_red, mean_green, mean_blue, median_red, median_green, median_blue, std_red, std_green, std_blue]

    return features

def process_images(root_directory, output_csv):
    # Open or create the CSV file for writing
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header to CSV file
        header = ["Label", "Filename", "Contrast", "Correlation", "Energy", "Homogeneity", "Area", "Perimeter", "Aspect Ratio",
                  "Mean Red", "Mean Green", "Mean Blue", "Median Red", "Median Green", "Median Blue",
                  "Std Red", "Std Green", "Std Blue"]
        writer.writerow(header)

        # Traverse through all subdirectories and process images
        label_counter = 1
        for root, dirs, files in os.walk(root_directory):
            for filename in files:
                if filename.endswith(".jpg"):
                    image_path = os.path.join(root, filename)
                    features = extract_features(image_path, label_counter, filename)
                    writer.writerow(features)
                    label_counter += 1

# Example usage:
root_directory =  r'C:\Users\hp\Desktop\mAJOR PROJECT\Aryvedic\major project codes\major project final codes\dataset2'  # Replace with your root directory
output_csv = 'features.csv'
process_images(root_directory, output_csv)
