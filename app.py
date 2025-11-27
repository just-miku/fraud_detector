import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model & scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset for demo purposes
data = pd.read_csv("creditcard.csv")

st.title("Fraud Detection AI System")
st.write("Demo of Random Forest-based fraud detection using real anonymized transactions.")

# Threshold slider
threshold = st.slider("Fraud Risk Threshold", 0.05, 0.95, 0.25)

# Button: Load random legitimate transaction
if st.button("Load Random Legit Transaction"):
    legit_data = data[data['Class'] == 0]
    sample = legit_data.sample(1)
    X_sample = sample.drop("Class", axis=1)
    y_true = sample["Class"].values[0]

    X_scaled = scaler.transform(X_sample)
    risk = model.predict_proba(X_scaled)[0][1]
    prediction = int(risk >= threshold)

    st.subheader(f"Fraud Risk Score: {risk:.4f}")
    if prediction == 1:
        st.error("⚠️ Model Prediction: FRAUD")
    else:
        st.success("✅ Model Prediction: LEGIT")

    st.info(f"Ground Truth: Legitimate Transaction")

# Button: Load random fraudulent transaction
if st.button("Load Random Fraud Transaction"):
    fraud_data = data[data['Class'] == 1]
    sample = fraud_data.sample(1)
    X_sample = sample.drop("Class", axis=1)
    y_true = sample["Class"].values[0]

    X_scaled = scaler.transform(X_sample)
    risk = model.predict_proba(X_scaled)[0][1]
    prediction = int(risk >= threshold)

    st.subheader(f"Fraud Risk Score: {risk:.4f}")
    if prediction == 1:
        st.error("⚠️ Model Prediction: FRAUD")
    else:
        st.success("✅ Model Prediction: LEGIT")

    st.warning("⚠️ Ground Truth: FRAUD Transaction")