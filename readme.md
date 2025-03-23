# 🏢 Employee Performance Score Predictor - Classification - Machine Learning Project #3

This is a **minimum viable product (MVP)** machine learning classification app that predicts employee performance scores using historical employee data. The goal is to demonstrate an end-to-end ML workflow from preprocessing, model selection, evaluation, to deployment using Streamlit.

## 🔍 Business Understanding

In HR analytics, predicting employee performance can support better decision-making in promotions, training, and employee retention strategies. This app helps:

- Identify likely high- or low-performing employees
- Understand the impact of salary, satisfaction, and work conditions
- Simulate performance outcomes based on user-defined inputs

This project is a simplified example of how classification models can be applied in a real-world business scenario.

## 🧠 ML Workflow Overview

1. **Data Cleaning & Preparation**
   - Removed unused or redundant columns
   - Handled duplicates and null values
   - Selected relevant features

2. **Feature Selection**
   - `Years_At_Company`
   - `Monthly_Salary`
   - `Overtime_Hours`
   - `Promotions`
   - `Employee_Satisfaction_Score`

3. **Preprocessing**
   - Scaled numeric features using `StandardScaler`
   - Split data into training/testing sets

4. **Model Training**
   - Trained and evaluated Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM)
   - Used `GridSearchCV` for hyperparameter tuning
   - Saved the best-performing model with `joblib`

5. **Deployment**
   - Built a Streamlit app for users to input data and receive predictions

## 📁 Files

- `app.py` → Streamlit app
- `employee.csv` → Dataset
- `model.pkl` → Best trained model
- `scaler.pkl` → Feature scaler
- `requirements.txt` → Dependencies list
- `README.md` → Project documentation (this file)

## 📦 Installation

```bash
pip install -r requirements.txt
