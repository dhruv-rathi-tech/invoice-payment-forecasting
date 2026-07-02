# Enterprise Invoice Payment Forecasting System

An end-to-end machine learning system to forecast invoice payment clearing timelines across large-scale financial transaction data.

---

## Overview

This project predicts the number of days it takes for a customer to clear an invoice after it is posted. Built on 50,000+ real business invoice records, it covers the full ML pipeline — data cleaning, feature engineering, model training, evaluation, and deployment — across two Streamlit applications.

---

## Project Structure

```
invoice_payment_forecasting/
│
├── train_model.py         # Pipeline dashboard (14-section walkthrough)
├── app.py                 # Live invoice prediction app
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** `dataset.csv` and generated `.pkl` artifact files (`model.pkl`, `rfe.pkl`, `encoders.pkl`, `feature_columns.pkl`) are not included in this repository due to size and privacy constraints. See setup instructions below.

---

## Applications

### 1. `train_model.py` — Pipeline Dashboard
A 14-section interactive dashboard that walks through the entire ML pipeline:
- Raw dataset exploration and EDA visualizations
- Data cleaning and preprocessing steps
- Date processing and feature engineering
- Encoding and transformation
- Train-test split and RFE feature selection
- Model training and comparison (Random Forest, Gradient Boosting, XGBoost)
- Final model summary and saved artifacts

### 2. `app.py` — Live Prediction App
A real-time inference interface where users input invoice details and receive:
- Predicted number of days to payment clearance
- Expected payment date
- On-time vs. delayed payment status
- Full feature breakdown sent to the model

---

## ML Pipeline Summary

| Stage | Details |
|---|---|
| Dataset | 50,000 invoice records, 19 raw features |
| Cleaning | Removed duplicates, dropped 6 identifier columns, removed null rows |
| Feature Engineering | 8 temporal features from date columns including due gap and calendar decomposition |
| Encoding | Label encoding, frequency grouping, one-hot encoding |
| Feature Selection | RFE with Random Forest base estimator → 15 features from 30+ |
| Models Evaluated | Random Forest, Gradient Boosting, XGBoost |
| Best Model | Random Forest Regressor |
| Test R² | 0.656 |
| Test RMSE | ~8 days |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/invoice_payment_forecasting.git
cd invoice_payment_forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place your `dataset.csv` file in the root of the project directory.

### 4. Run the pipeline dashboard
This will train the models and generate the `.pkl` artifact files.
```bash
streamlit run train_model.py
```

### 5. Run the prediction app
Only after step 4 has completed and `.pkl` files are generated.
```bash
streamlit run app.py
```

---

## Tech Stack

- **Python 3.10+**
- **Pandas, NumPy** — data manipulation
- **Scikit-learn** — preprocessing, RFE, Random Forest, Gradient Boosting
- **XGBoost** — gradient boosting regressor
- **Matplotlib, Seaborn** — visualizations
- **Streamlit** — interactive web applications
- **Joblib** — model serialization

---

## Author

Dhruv — Electronics and Computer Engineering, VIT Chennai
