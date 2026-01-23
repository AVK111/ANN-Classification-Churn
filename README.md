# Customer Churn Prediction using Artificial Neural Network (ANN)

## Overview
This project predicts whether a customer is likely to churn (leave a service) based on various business-related features. An Artificial Neural Network (ANN) is used to learn patterns from structured data and estimate churn probability.

The project demonstrates a complete end-to-end machine learning pipeline including data preprocessing, model training, evaluation, and deployment using Streamlit.

---

## Dataset
The dataset contains the following features:

- CreditScore  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- NumOfProducts  
- HasCrCard  
- IsActiveMember  
- EstimatedSalary  

Target variable:
- Exited (0 = Not churned, 1 = Churned)

---

## Data Preprocessing
The following preprocessing steps were applied:

- Removed irrelevant identifiers: RowNumber, CustomerId, Surname  
- Label Encoding for Gender  
- One-Hot Encoding for Geography  
- Feature Scaling using StandardScaler  

All encoders and the scaler were saved using `pickle` for reuse during inference.

---

## Model Architecture
The Artificial Neural Network consists of:

- Input layer based on number of features  
- Two hidden layers with ReLU activation  
- Output layer with Sigmoid activation  

Loss Function: Binary Cross-Entropy  
Optimizer: Adam  

---

## Training Strategy
- Train-test split applied  
- Validation split used during training  
- Early Stopping to prevent overfitting  
- TensorBoard for monitoring training performance  

---

## Deployment
The trained model is deployed using Streamlit.  
Users can enter customer details and receive churn probability in real time.

---

## Key Learnings
- Importance of preprocessing structured data  
- Effect of feature scaling on neural networks  
- Difference between training and inference pipelines  
- Building complete ML systems from scratch  

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas, NumPy  
- Streamlit  
