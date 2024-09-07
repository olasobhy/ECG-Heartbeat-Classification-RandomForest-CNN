# ECG-Heartbeat-Classification-Using-RandomForest-CNN
## Project Overview
This project focuses on classifying ECG (Electrocardiogram) heartbeats into five categories using both Random Forest and Convolutional Neural Network (CNN) models. The dataset used is from the MIT-BIH arrhythmia database, which contains labeled ECG recordings. We aim to build models that classify heartbeats into one of the following categories:

Normal
Supraventricular
Ventricular
Fusion
Unknown
## Dataset
The dataset is publicly available on Kaggle: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

## Files Used:
mitbih_train.csv: Training data with 187 features and 1 target label
mitbih_test.csv: Test data for model evaluation
Each record contains 187 data points representing an ECG signal over time, with the last column being the target label indicating the heartbeat category.

## Dataset Statistics:
Normal Beats (0): 72,471
<br>
Supraventricular Beats (1): 2,223
<br>
Ventricular Beats (2): 5,788
<br>
Fusion Beats (3): 641
<br>
Unknown Beats (4): 6,431
<br>
## Model Implementation
### 1. Random Forest Classifier
A Random Forest model is used to classify the ECG data. To address class imbalance in the dataset, we used SMOTE (Synthetic Minority Oversampling Technique) to balance the training set.

## Key Steps:
Applied SMOTE for class balancing.
Split the data into training and test sets.
Achieved an accuracy of 99.65%.

### 2. Convolutional Neural Network (CNN)
Implemented a CNN model to capture the time-series nature of ECG signals. The model consists of a Conv1D layer, MaxPooling1D, and a Dense output layer. We used sparse_categorical_crossentropy as the loss function and Adam optimizer.

CNN Architecture:
Input: 1D signal of shape (187, 1)
<br>
Conv1D layer with 64 filters and kernel size of 3
<br>
MaxPooling1D layer with pool size 2
<br>
Flatten layer
<br>
Dense output layer with softmax activation (for 5 categories)
<br>
