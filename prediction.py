import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the VGG16 model with the specified weights file
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv2').output)

# Load the Random Forest classifier
rf_classifier = joblib.load('model_random_forest_dataset2.pkl')

# Function to extract features using VGG16
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Path to the directory containing subfolders with images
dataset_dir = r'C:\Users\hp\Desktop\mAJOR PROJECT\Aryvedic\major project codes\major project final codes\dataset2'

# Loop through each subfolder (class) in the dataset directory
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    # Loop through each image file in the subfolder
    for img_filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_filename)
        
        # Extract features from the image
        image_features = extract_features(img_path)
        prediction_features = image_features.reshape(1, -1)
        
        # Predict using the Random Forest classifier
        prediction = rf_classifier.predict(prediction_features)
        
        # Display the prediction
        print(f"Image: {img_filename}, Prediction: {prediction}, Class: {class_name}")
