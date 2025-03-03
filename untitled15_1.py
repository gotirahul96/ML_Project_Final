
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load saved objects
optimized_rf = pickle.load(open("optimized_rf.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
numerical_features = pickle.load(open("features.pkl", "rb"))  # Load original feature names

# Ensure user inputs match model features
st.title("Credit Default Prediction App")
st.write("This app predicts if a customer will default next month based on financial data.")

# Collect user inputs
credit_limit = st.number_input("Credit Limit", min_value=0, value=50000)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bill_amt = st.number_input("Last Month's Bill Amount", min_value=0, value=10000)
pay_amt = st.number_input("Last Month's Payment Amount", min_value=0, value=5000)
pay_status = st.selectbox("Repayment Status Last Month", ["No Delay", "Delayed by 1 Month", "Delayed by 2 Months"])

# Convert categorical value PAY_1 before inserting into DataFrame
pay_status_mapping = {
    "No Delay": 0,
    "Delayed by 1 Month": 1,
    "Delayed by 2 Months": 2
}
pay_status_encoded = pay_status_mapping[pay_status]  # Convert to integer

# Calculate ratios (Only if these were used in training)
pay_to_bill_ratio = pay_amt / (bill_amt + 1e-6)
credit_utilization_ratio = bill_amt / (credit_limit + 1e-6)

# Ensure correct feature order and match training data
input_data = pd.DataFrame(columns=numerical_features)  # Match feature order

# Assign user inputs to the correct columns
input_data.loc[0] = [0] * len(numerical_features)  # Initialize all features with zero
if 'LIMIT_BAL' in numerical_features:
    input_data['LIMIT_BAL'] = credit_limit
if 'AGE' in numerical_features:
    input_data['AGE'] = age
if 'BILL_AMT1' in numerical_features:
    input_data['BILL_AMT1'] = bill_amt  # Ensure the correct column name
if 'PAY_AMT1' in numerical_features:
    input_data['PAY_AMT1'] = pay_amt
if 'PAY_TO_BILL_RATIO' in numerical_features:
    input_data['PAY_TO_BILL_RATIO'] = pay_to_bill_ratio
if 'CREDIT_UTILIZATION_RATIO' in numerical_features:
    input_data['CREDIT_UTILIZATION_RATIO'] = credit_utilization_ratio
if 'PAY_1' in numerical_features:
    input_data['PAY_1'] = pay_status_encoded  # ✅ Fixed mapping issue

# Ensure all columns are numeric
input_data = input_data.astype(float)

# Apply feature scaling
input_data_scaled = scaler.transform(input_data)

# Make prediction
if st.button("Predict Default Risk"):
    probability = optimized_rf.predict_proba(input_data_scaled)[0][1]
    
    # Set a decision threshold (default is 0.5, adjust if necessary)
    threshold = 0.5
    prediction = 1 if probability >= threshold else 0

    if prediction == 1:
        st.error(f"⚠ High Risk of Default! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Default. Probability: {probability:.2f}")
