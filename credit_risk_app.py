# Install required libraries
# pip install pandas numpy scikit-learn matplotlib seaborn streamlit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
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
        'esg_score': np.random.uniform(40, 80, num_samples),  # ESG score (from 0 to 100)
        'num_bounced_payments': np.random.poisson(1, num_samples),  # Number of bounced payments
        'loan_default': np.random.binomial(1, 0.2, num_samples)  # Loan default (0 or 1)
    })
    data['dti'] = data['existing_debt'] / data['avg_income']  # Debt-to-Income Ratio
    data['ltv'] = data['loan_amount'] / (data['avg_income'] * 12)  # Loan-to-Value Ratio
    return data

# Section 2: Exploratory Data Analysis (EDA)
def perform_eda(data):
    st.title("Exploratory Data Analysis")
    
    # Histograms
    st.subheader("Histograms")
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    sns.histplot(data['avg_income'], kde=True, ax=axs[0, 0]).set_title('Average Income')
    sns.histplot(data['existing_debt'], kde=True, ax=axs[0, 1]).set_title('Existing Debt')
    sns.histplot(data['loan_amount'], kde=True, ax=axs[0, 2]).set_title('Loan Amount')
    sns.histplot(data['esg_score'], kde=True, ax=axs[1, 0]).set_title('ESG Score')
    sns.histplot(data['num_bounced_payments'], kde=True, ax=axs[1, 1]).set_title('Number of Bounced Payments')
    sns.histplot(data['loan_default'], kde=True, ax=axs[1, 2]).set_title('Loan Default')
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Section 3: Model Training
def train_model(data):
    st.title("Model Training")
    
    # Split the data
    X = data.drop('loan_default', axis=1)
    y = data['loan_default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.text(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.2f}")
    
    return model, X_test

# Section 4: Streamlit Application
def main():
    data = generate_data()
    st.sidebar.title("Loan Risk Analysis")
    
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.write(data)
    
    if st.sidebar.checkbox("Perform EDA"):
        perform_eda(data)
    
    if st.sidebar.checkbox("Train Model"):
        model, X_test = train_model(data)
    
    st.sidebar.title("Predict Loan Risk")
    avg_income = st.sidebar.slider("Average Income", 10000, 100000, 40000)
    existing_debt = st.sidebar.slider("Existing Debt", 5000, 50000, 20000)
    loan_amount = st.sidebar.slider("Loan Amount", 10000, 100000, 50000)
    esg_score = st.sidebar.slider("ESG Score", 0, 100, 60)
    num_bounced_payments = st.sidebar.slider("Number of Bounced Payments", 0, 10, 1)
    
    if st.sidebar.button("Predict"):
        input_data = pd.DataFrame({
            'avg_income': [avg_income],
            'existing_debt': [existing_debt],
            'loan_amount': [loan_amount],
            'esg_score': [esg_score],
            'num_bounced_payments': [num_bounced_payments],
            'dti': [existing_debt / avg_income],
            'ltv': [loan_amount / (avg_income * 12)]
        })
        prediction = model.predict(input_data)[0]
        st.subheader("Prediction")
        st.write("Loan Default" if prediction == 1 else "No Loan Default")
    
if __name__ == "__main__":
    main()
