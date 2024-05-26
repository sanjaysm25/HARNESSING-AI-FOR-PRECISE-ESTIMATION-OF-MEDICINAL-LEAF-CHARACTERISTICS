import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import csv

# Directory containing your dataset
dataset_dir = r'C:\Users\hp\Desktop\mAJOR PROJECT\Aryvedic\major project codes\major project final codes\dataset2'

# Create a list to store annotations
annotations = []

# Iterate through each image in the dataset directory
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        # Get the full path to the image
        image_path = os.path.join(root, file)

        # Read the image
        img = cv2.imread(image_path)

        # Resize the image for consistency
        img = cv2.resize(img, (224, 224))

        # Flatten the image to use as feature vector
        features = img.flatten().reshape(-1, 3)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(features)

        # Assign the cluster label as the category
        category = f'Cluster_{kmeans.labels_[0]}'

        # Append the annotation to the list
        annotations.append({'image_file': image_path, 'label': category})

# Define the path for the CSV file
csv_file_path = 'labels.csv'

# Write annotations to CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    fieldnames = ['image_file', 'label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write header
    writer.writeheader()
    
    # Write data
    for annotation in annotations:
        writer.writerow(annotation)

print(f"Labeling completed. CSV file saved to {csv_file_path}")
