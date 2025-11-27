import streamlit as st
import numpy as np
import joblib

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Fraud Detection AI System")

amount = st.number_input("Transaction Amount", 0.0)
time = st.number_input("Transaction Time", 0.0)

features = [time, amount] + [0] * 28  # dummy V-features

if st.button("Check Fraud Risk"):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    risk = model.predict_proba(features_scaled)[0][1]

    st.subheader(f"Fraud Risk Score: {risk:.4f}")

    if risk > 0.5:
        st.error("⚠️ High Fraud Risk")
    else:
        st.success("✅ Transaction Appears Legitimate")