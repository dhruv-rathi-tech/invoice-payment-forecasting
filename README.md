# PayCast – Enterprise Invoice Payment Forecasting

**PayCast** is an end-to-end machine learning system to forecast invoice payment clearing timelines across large-scale financial transaction data.

---

## Live Demo

🔗 https://invoicepaymentforecasting.streamlit.app

---

## Overview

This project predicts the number of days required for a customer to clear an invoice after it is posted.

Built using **50,000+ real-world business invoice records**, the project demonstrates the complete machine learning workflow—from data preprocessing and feature engineering to model training, evaluation, and deployment.

The repository contains two Streamlit applications:

- **Training Dashboard** – explains the complete ML pipeline
- **Prediction App** – performs real-time invoice payment prediction using a pre-trained model

---

## Project Structure

```text
paycast/
│
├── app.py                     # Live prediction application
├── train_model.py             # End-to-end ML pipeline dashboard
├── main.py                    # Initial EDA and preprocessing experiments
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── Usecase.pdf
└── problem_statement.pdf
```

---

## Applications

### 1. Training Dashboard (`train_model.py`)

A comprehensive 14-section interactive dashboard demonstrating the entire machine learning pipeline:

- Dataset exploration
- Exploratory Data Analysis (EDA)
- Data cleaning
- Feature engineering
- Date processing
- Label Encoding
- One-Hot Encoding
- Train/Test split
- Recursive Feature Elimination (RFE)
- Model training
- Hyperparameter configuration
- Model comparison
- Performance evaluation
- Model artifact generation

---

### 2. Prediction Application (`app.py`)

A production-style Streamlit application that loads the pre-trained model and predicts invoice payment timelines.

Users can:

- Select existing or enter new customer information
- Enter invoice details
- Predict invoice clearing time
- Estimate payment date
- Identify on-time or delayed payments

The prediction app performs inference only and **does not retrain the model**, resulting in significantly faster startup.

---

## Machine Learning Pipeline

| Stage | Details |
|--------|---------|
| Dataset | 50,000 invoice records |
| Initial Features | 19 raw features |
| Data Cleaning | Duplicate removal, null handling, irrelevant column removal |
| Feature Engineering | Calendar features, due gap, payment duration |
| Encoding | Label Encoding, Frequency Encoding, One-Hot Encoding |
| Feature Selection | Recursive Feature Elimination (15 selected features) |
| Models Evaluated | Random Forest, Gradient Boosting, XGBoost |
| Final Model | Random Forest Regressor |
| Test R² Score | **0.656** |
| Test RMSE | **~8 days** |

---

## Saved Model Artifacts

The training dashboard generates the following files:

| File | Purpose |
|------|----------|
| model.pkl | Trained Random Forest model |
| encoders.pkl | Label encoders for customer features |
| feature_columns.pkl | Original feature ordering |
| selected_features.pkl | Final 15 features selected using RFE |
0
---

## Setup

### Clone the repository

```bash
git clone https://github.com/dhruv-rathi-tech/invoice-payment-forecasting.git
cd invoice-payment-forecasting
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

### Add dataset

Place

```
dataset.csv
```

inside the project root.

---

### Generate model artifacts

Run

```bash
streamlit run train_model.py
```

This will:

- preprocess the dataset
- engineer features
- train all models
- evaluate performance
- generate the required `.pkl` files

---

### Launch the prediction app

After the artifacts have been generated,

```bash
streamlit run app.py
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## Future Improvements

- REST API using FastAPI
- Docker deployment
- Batch invoice prediction
- Explainable AI (SHAP)
- Database integration
- Automated model retraining pipeline

---

## Author

**Dhruv Rathi**