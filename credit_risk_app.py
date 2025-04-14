import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Set Streamlit config
st.set_page_config(page_title="Loan Risk Analyzer", layout="centered")
sns.set(style="whitegrid")

# Section 1: Data Generation
@st.cache_data
def generate_data(num_samples=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'avg_income': np.random.normal(40000, 10000, num_samples),  
        'existing_debt': np.random.normal(20000, 5000, num_samples),  
        'loan_amount': np.random.normal(50000, 15000, num_samples),  
        'esg_score': np.random.uniform(40, 80, num_samples),  
        'num_bounced_payments': np.random.poisson(1, num_samples),  
    })
    data['dti'] = data['existing_debt'] / data['avg_income']  
    data['ltv'] = data['loan_amount'] / (data['avg_income'] * 12)  

    # Custom logic for loan default generation
    prob_default = (
        0.15 +
        0.4 * (data['dti'] > 0.6).astype(int) +
        0.3 * (data['ltv'] > 0.8).astype(int) +
        0.2 * (data['num_bounced_payments'] > 1).astype(int)
    )
    data['loan_default'] = np.random.binomial(1, np.clip(prob_default, 0, 1))
    return data

# Section 2: EDA
def perform_eda(data):
    st.subheader("Exploratory Data Analysis")

    # Histograms
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    sns.histplot(data['avg_income'], kde=True, ax=axs[0, 0]).set_title('Average Income')
    sns.histplot(data['existing_debt'], kde=True, ax=axs[0, 1]).set_title('Existing Debt')
    sns.histplot(data['loan_amount'], kde=True, ax=axs[0, 2]).set_title('Loan Amount')
    sns.histplot(data['esg_score'], kde=True, ax=axs[1, 0]).set_title('ESG Score')
    sns.histplot(data['num_bounced_payments'], kde=True, ax=axs[1, 1]).set_title('Bounced Payments')
    sns.histplot(data['loan_default'], kde=False, ax=axs[1, 2]).set_title('Loan Default (0 = No, 1 = Yes)')
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Section 3: Model Training
@st.cache_resource
def train_model(data):
    X = data.drop('loan_default', axis=1)
    y = data['loan_default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return model, classification_report(y_test, y_pred), roc_auc_score(y_test, y_proba)

# Section 4: Main App
def main():
    st.title("💳 Loan Default Risk Prediction App")

    data = generate_data()
    model, clf_report, auc_score = train_model(data)

    with st.expander("📊 Show Raw Data"):
        st.write(data.head())

    if st.checkbox("📈 Perform EDA"):
        perform_eda(data)

    st.subheader("⚙️ Input Financial Indicators")

    avg_income = st.slider("Average Monthly Income (INR)", 10000, 100000, 40000, step=5000)
    existing_debt = st.slider("Existing Debt (INR)", 0, 70000, 20000, step=5000)
    loan_amount = st.slider("Loan Amount Requested (INR)", 10000, 200000, 50000, step=5000)
    esg_score = st.slider("ESG Score", 0, 100, 60, step=5)
    num_bounced_payments = st.slider("Bounced Payments (past year)", 0, 5, 1)

    if st.button("🔍 Predict Loan Default Risk"):
        dti = existing_debt / avg_income
        ltv = loan_amount / (avg_income * 12)
        
        input_df = pd.DataFrame([{
            'avg_income': avg_income,
            'existing_debt': existing_debt,
            'loan_amount': loan_amount,
            'esg_score': esg_score,
            'num_bounced_payments': num_bounced_payments,
            'dti': dti,
            'ltv': ltv
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Risk: Likely to Default (Probability = {probability:.2f})")
        else:
            st.success(f"✅ Low Risk: Likely to Repay (Probability = {probability:.2f})")

    st.markdown("---")
    st.subheader("📋 Model Evaluation Metrics")
    st.code(clf_report)
    st.write(f"ROC AUC Score: **{auc_score:.2f}**")

if __name__ == "__main__":
    main()
