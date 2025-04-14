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
    data = pd.DataFrame({
        'avg_income': np.random.normal(40000, 10000, num_samples),  # Monthly income (INR)
        'existing_debt': np.random.normal(20000, 5000, num_samples),  # Existing debt (INR)
        'loan_amount': np.random.normal(50000, 15000, num_samples),  # Loan amount (INR)
        'esg_score': np.random.uniform(40, 80, num_samples),  # ESG score
    })
    
    # Calculate DTI and LTV
    data['dti'] = data['existing_debt'] / data['avg_income']
    data['ltv'] = data['loan_amount'] / (data['avg_income'] * 12)  # Assuming monthly income is annualized
    
    # Define default label based on DTI and LTV thresholds
    data['default'] = np.where((data['dti'] > 0.4) & (data['ltv'] > 0.8), 1, 0)
    
    return data

# Section 2: Model Training
def train_model(data):
    X = data[['avg_income', 'existing_debt', 'loan_amount', 'esg_score', 'dti', 'ltv']]
    y = data['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    
    return model

# Section 3: Streamlit App
def main():
    st.title("Loan Default Prediction")
    
    # Generate and display data
    data = generate_data()
    st.write("Generated Data:")
    st.write(data.head())
    
    # Train model
    model = train_model(data)
    
    # User input
    st.sidebar.header("User Input Parameters")
    def user_input_features():
        avg_income = st.sidebar.number_input("Average Income (INR)", 20000, 80000, 40000)
        existing_debt = st.sidebar.number_input("Existing Debt (INR)", 0, 50000, 20000)
        loan_amount = st.sidebar.number_input("Loan Amount (INR)", 10000, 100000, 50000)
        esg_score = st.sidebar.slider("ESG Score", 0, 100, 60)
        
        dti = existing_debt / avg_income
        ltv = loan_amount / (avg_income * 12)
        
        data = {
            'avg_income': avg_income,
            'existing_debt': existing_debt,
            'loan_amount': loan_amount,
            'esg_score': esg_score,
            'dti': dti,
            'ltv': ltv
        }
        features = pd.DataFrame(data, index=[0])
        return features
    
    input_df = user_input_features()
    st.write("User Input Parameters:")
    st.write(input_df)
    
    # Predict default
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]
    
    st.write("Prediction (0: No Default, 1: Default):", int(prediction[0]))
    st.write("Prediction Probability:", prediction_proba[0])

if __name__ == "__main__":
    main()
