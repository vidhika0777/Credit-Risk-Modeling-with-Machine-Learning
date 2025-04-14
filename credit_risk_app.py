# Install required libraries
# pip install pandas numpy scikit-learn matplotlib seaborn shap streamlit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import streamlit as st

# Set plot style
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
        'num_board_members': np.random.randint(5, 15, num_samples),  # Number of board members
        'company_age': np.random.randint(1, 50, num_samples),  # Age of the company (years)
        'loan_default': np.random.binomial(1, 0.2, num_samples)  # Loan default (binary)
    })

    # Calculate additional features
    data['dti_ratio'] = data['existing_debt'] / data['avg_income']  # Debt-to-Income Ratio
    data['ltv_ratio'] = data['loan_amount'] / (data['avg_income'] * 12)  # Loan-to-Value Ratio (annual income)

    return data

# Section 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis(data):
    st.subheader("Exploratory Data Analysis")
    st.write(data.describe())

    # Pairplot
    st.subheader("Pairplot")
    sns.pairplot(data)
    st.pyplot()

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot()

# Section 3: Model Training
def model_training(data):
    st.subheader("Model Training")

    # Define features and target
    X = data.drop('loan_default', axis=1)
    y = data['loan_default']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write("ROC AUC Score:", roc_auc_score(y_test, y_pred))

    return model, X_test

# Section 4: SHAP Analysis
def shap_analysis(model, X_test):
    st.subheader("SHAP Analysis")

    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    st.write("Feature Importance:")
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    st.pyplot()
    shap.summary_plot(shap_values, X_test)
    st.pyplot()

# Main function to run the Streamlit app
def main():
    st.title("Loan Risk Assessment")

    # Generate data
    data = generate_data()
    st.write("Generated Data:")
    st.write(data.head())

    # Exploratory Data Analysis
    exploratory_data_analysis(data)

    # Model Training
    model, X_test = model_training(data)

    # SHAP Analysis
    shap_analysis(model, X_test)

if __name__ == "__main__":
    main()
