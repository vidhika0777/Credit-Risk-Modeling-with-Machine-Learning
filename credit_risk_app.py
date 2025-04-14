# Install required libraries
# pip install pandas numpy scikit-learn matplotlib seaborn streamlit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import streamlit as st

# Set plot style for EDA
sns.set(style="whitegrid")

# Section 1: Data Generation
# Simulate data with additional features like DTI (Debt-to-Income Ratio), LTV (Loan-to-Value Ratio)
def generate_data(num_samples=1000):
    np.random.seed(42)
    avg_income = np.random.normal(40000, 10000, num_samples)  # Monthly income (INR)
    existing_debt = np.random.normal(20000, 5000, num_samples)  # Existing debt (INR)
    loan_amount = np.random.normal(50000, 15000, num_samples)  # Loan amount (INR)
    esg_score = np.random.uniform(40, 80, num_samples)  # ESG score

    # Calculate DTI and LTV
    dti = existing_debt / avg_income
    ltv = loan_amount / (avg_income * 12)  # Assuming yearly income

    # Define default based on thresholds
    default = (dti > 0.4) | (ltv > 0.8)  # Arbitrary thresholds for demonstration

    data = pd.DataFrame({
        'avg_income': avg_income,
        'existing_debt': existing_debt,
        'loan_amount': loan_amount,
        'esg_score': esg_score,
        'dti': dti,
        'ltv': ltv,
        'default': default.astype(int)
    })

    return data

# Section 2: Model Training
def train_model(data):
    features = ['avg_income', 'existing_debt', 'loan_amount', 'esg_score', 'dti', 'ltv']
    target = 'default'

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, y_pred, y_pred_proba

# Section 3: Model Evaluation
def evaluate_model(y_test, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, roc_auc, report, cm

# Section 4: Streamlit App
def main():
    st.title("Loan Default Prediction")

    # Generate data
    num_samples = st.sidebar.slider("Number of samples", 100, 10000, 1000)
    data = generate_data(num_samples)

    st.write("### Generated Data")
    st.write(data.head())

    # Train model
    model, X_test, y_test, y_pred, y_pred_proba = train_model(data)

    # Evaluate model
    accuracy, roc_auc, report, cm = evaluate_model(y_test, y_pred, y_pred_proba)

    st.write("### Model Evaluation")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"ROC AUC Score: {roc_auc:.2f}")
    st.write("Classification Report:")
    st.text(report)
    st.write("Confusion Matrix:")
    st.write(cm)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    st.write("### Feature Importance")
    st.bar_chart(feature_importance.set_index('feature'))

    # User input for prediction
    st.write("### Predict Loan Default")
    avg_income = st.number_input("Average Income (INR)", value=40000)
    existing_debt = st.number_input("Existing Debt (INR)", value=20000)
    loan_amount = st.number_input("Loan Amount (INR)", value=50000)
    esg_score = st.number_input("ESG Score", value=60)

    # Calculate DTI and LTV for user input
    dti = existing_debt / avg_income
    ltv = loan_amount / (avg_income * 12)

    user_data = pd.DataFrame({
        'avg_income': [avg_income],
        'existing_debt': [existing_debt],
        'loan_amount': [loan_amount],
        'esg_score': [esg_score],
        'dti': [dti],
        'ltv': [ltv]
    })

    st.write("User Data:")
    st.write(user_data)

    if st.button("Predict"):
        user_pred = model.predict(user_data)
        user_pred_proba = model.predict_proba(user_data)[:, 1]

        st.write(f"Prediction: {'Default' if user_pred[0] == 1 else 'No Default'}")
        st.write(f"Prediction Probability: {user_pred_proba[0]:.2f}")

if __name__ == "__main__":
    main()



