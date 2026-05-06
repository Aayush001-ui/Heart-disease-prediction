# Heart Disease Prediction Web App

A machine learning web application that predicts the risk of heart disease based on patient medical data.

This project demonstrates an end-to-end ML pipeline including model training, evaluation, deployment using Flask, and model explainability using SHAP with visualizations.

## Features

* Interactive web interface for user input
* Predicts heart disease risk using a trained ML model
* Displays prediction confidence score
* Shows medical reference values for interpretation
* Includes **SHAP-based explainability with visualizations**
* End-to-end integration (ML + Backend + Frontend).

## Tech Stack

* Python
* Flask
* Scikit-learn
* NumPy, Pandas
* SHAP (Explainable AI)
* Matplotlib, Seaborn
* HTML, CSS

## Machine Learning Workflow

1. Data preprocessing and feature selection
2. Train-test split (stratified sampling)
3. Feature scaling using StandardScaler
4. Model training:

   * Logistic Regression
   * Random Forest (selected as best model)
5. Model evaluation:

   * Accuracy
   * ROC-AUC
   * Precision, Recall, F1-score
6. Model saved using Joblib
7. Integrated with Flask for real-time predictions


## Explainability (SHAP)

This project uses SHAP (SHapley Additive exPlanations) to interpret model predictions.

* Applied SHAP TreeExplainer on Random Forest model
* Generated **SHAP summary plot** to understand global feature importance
* Visualized how each feature contributes to heart disease prediction
* Helps in making the model transparent and interpretable

##  How to Run Locally

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train the model

python model_training.py

### 3. Run the Flask app

python app.py

### 4. Open in browser

http://127.0.0.1:5000

## Project Structure

heart-disease-prediction/
├── app.py
├── model_training.py
├── analysis/
│   └── analysis.py
├── templates/
│   └── index.html
├── static/
├── requirements.txt
├── README.md
├── .gitignore

---

## Model Performance

* Accuracy: ~92%
* Balanced class handling
* Evaluated using multiple metrics (AUC, Precision, Recall, F1-score)

## Future Improvements

* Integrate SHAP visualization directly into UI
* Deploy application online
* Improve UI/UX
* Add detailed patient-level explanation

## Author

Aayush Kasaudhan
