# HARNESSING-AI-FOR-PRECISE-ESTIMATION-OF-MEDICINAL-LEAF-CHARACTERISTICS

Base Papers:
Paper Title: The Classification of Medicinal Plant Leaves Based on Multispectral and Texture Feature Using Machine Learning Approach

Methodology: Utilizes a machine learning approach incorporating multispectral and texture features for classification.
Scope for Improvement: Enhance classification accuracy using advanced machine learning techniques and improve feature extraction methods.
Improvement Done: Achieved an improved classification accuracy of 92% through enhancements in machine learning models.

Paper Title: Automatic Recognition of Medicinal Plants using Machine Learning Techniques

Methodology: Employs a Random Forest Classifier, feature extraction, and SVM classification for automatic recognition.
Scope for Improvement: Address lighting and leaf orientation variations, increase dataset diversity through data augmentation, and handle overfitting.
Improvement Done: Expanded the dataset to 30 species, implemented data augmentation, and mitigated overfitting issues.

Paper Title: A Convolutional Neural Network-driven Computer Vision System toward Identification of Species and Maturity Stage of Medicinal Leaves

Methodology: Utilizes APRS, convolutional neural networks (CNN), and Geographic Information Systems (GIS) technology for leaf identification.
Scope for Improvement: Enhance feature extraction for better discrimination, incorporate domain knowledge, and address overfitting.
Improvement Done: Improved feature extraction techniques and resolved overfitting concerns.

Paper Title: Deep Convolutional Neural Network-based Plant Species Recognition through Features of Leaf

Methodology: Utilizes a Multilayer Perceptron (MLP) classifier for plant species recognition.
Scope for Improvement: Enhance model interpretability and handle intra-class variations for better class differentiation.
Improvement Done: Enhanced class differentiation and potentially improved model interpretability.

Preprocessing of the Dataset
STEP 1 :- Standardization1.py reads images from a directory, resizes them to (224, 224), normalizes pixel values, and overwrites the originals.

STEP 2:- labelling .py processes images in a directory, performs clustering on flattened pixel values, assigns labels, and saves annotations in a CSV file.

STEP 3:-feature extraction .py extracts various features (texture, shape, and color) from leaf images in a specified directory and writes the results to a CSV file for further analysis. Features include contrast, correlation, energy, homogeneity, area, perimeter, aspect ratio, mean values, median values, and standard deviations for red, green, and blue color channels.

STEP 4:-feature selections .py select the features required for the prepossessing

STEP 5:-pre processing .py uses TensorFlow and Keras to build a medicinal leaf image classification model with data augmentation. It defines a simple LSTM-based architecture, compiles the model, and trains it on augmented image data with checkpoints and early stopping. Optionally, it loads a pre-trained

Dataset Augmentation
dataset_preparation_with_augmentation.py code

This Python script augments a dataset by applying random transformations such as rotation, shifting, and flipping to the original images and saves the augmented dataset in a target directory, allowing for increased diversity and size of the dataset for training machine learning models.

Training VGG16 Model
train_vgg_dataset2.py code

This script utilizes transfer learning with VGG16 for image classification, augmenting data, training a custom head, and evaluating performance through metrics visualization and prediction generation.

The steps can be summarized as follows:
Data preparation and augmentation using ImageDataGenerator.

Loading the VGG16 network with pre-trained weights and constructing a custom classification head.

Compiling and training the model.

Saving the trained model and visualizing training history.

Evaluating the model's performance on the validation set and generating predictions.

Plotting evaluation metrics including accuracy, loss, classification report, and confusion matrix.

Training Random Forest Model
train_randomforest_ensemble.py code

This script uses transfer learning with VGG16 to extract features from images, trains a Random Forest classifier, and evaluates its performance on a dataset, achieving an accuracy score. Finally, it saves the trained model.

The steps can be summarized as follows: Loads the VGG16 model with pre-trained ImageNet weights.

Defines a function to extract features using VGG16 from image paths.

Processes images in the dataset directory, extracting features and corresponding labels.

Splits the dataset into training and testing sets.

Trains a Random Forest classifier on the extracted features.

Evaluates the classifier's performance on the testing set.

Saves the trained Random Forest model for future use.

Prediction
prediction.py code

This script utilizes a pre-trained VGG16 model to extract features from images in a dataset. Then, it employs a Random Forest classifier to predict the class of each image based on the extracted features.

App
app.py code

This Streamlit app allows users to upload leaf images for classification and analysis. It uses a pre-trained VGG16 model to extract features and a Random Forest classifier for prediction. The predictions from both models are compared, and additional information about the predicted class is displayed if available.

** The steps can be summarized as follows:**

Import Libraries: Necessary libraries such as os, streamlit, numpy, tensorflow, joblib, and PIL are imported.

Load Models: Pre-trained VGG16 model and Random Forest classifier are loaded.

Define Functions: Functions for feature extraction using VGG16, prediction using Random Forest, and image preprocessing are defined.

Create Streamlit App: A Streamlit web application with a title and file uploader is created.

Check and Predict: Uploaded image is checked, and predictions are made using both VGG16 and Random Forest models.

Display Predictions: Display the predicted class and detailed information. Majority voting is used to display the prediction if available; otherwise, the prediction with the highest accuracy algorithm is displayed.

Error Handling: If the uploaded image is not recognized, an error message is displayed.

User Interface: Provide a user-friendly interface for users to interact with and obtain precise estimations of medical leaf characteristics.

-----------------------------------THANK YOU--------------------------------------
