Wine Quality Predictor (ML)

This project predicts the quality of red and white wine using Machine Learning with a Neural Network built in TensorFlow/Keras.
It uses two datasets — one for red wine and one for white wine — and allows predictions based on user-provided chemical properties.

Project Structure:

wine-quality-predictor-ml/
wine_train.py # Trains models for red and white wine 
wine_predict.py # Takes user input & predicts wine quality
winequality-red.csv # Dataset for red wine 
winequality-white.csv# Dataset for white wine 
wine_models_paths.save # Stores paths to models & scalers 
scaler_red.save # StandardScaler for red wine 
scaler_white.save # StandardScaler for white wine
README.md # This file

How to Run:

Train the Models 
Run:python wine_train.py 
This will: Train two separate models (red & white wine), Save the models and scalers, Create a single wine_models_paths.save file for easy loading.

Predict Wine Quality
Run:python wine_predict.py
You will be prompted to: Select wine type (red or white), Enter values for each chemical property, The program will output the predicted quality score.

Dataset: 
The datasets are from the UCI Machine Learning Repository and contain the following features: Fixed acidity Volatile acidity Citric acid Residual sugar Chlorides Free sulfur dioxide Total sulfur dioxide Density pH Sulphates Alcohol Target variable: Quality (an integer score between 0–10)

Requirements: 
Install pandas,scikit-learn,tensorflow and joblib

Example Output: 
Enter wine type (red/white): red Enter fixed acidity: 7.4 Enter volatile acidity: 0.70 ... Predicted quality: 5.82
