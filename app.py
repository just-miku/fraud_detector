import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset for realistic testing
data = pd.read_csv("creditcard.csv")

st.title("Fraud Detection AI System")

st.write("This demo uses real transaction samples from the dataset.")

# Select random transaction
if st.button("Load Random Transaction"):
    sample = data.sample(1)

    X_sample = sample.drop("Class", axis=1)
    y_true = sample["Class"].values[0]

    X_scaled = scaler.transform(X_sample)

    risk = model.predict_proba(X_scaled)[0][1]
    prediction = int(risk >= 0.25)

    st.subheader(f"Fraud Risk Score: {risk:.4f}")

    if prediction == 1:
        st.error("⚠️ Model Prediction: FRAUD")
    else:
        st.success("✅ Model Prediction: LEGIT")

    if y_true == 1:
        st.warning("⚠️ Ground Truth: This transaction IS actually fraud.")
    else:
        st.info("✅ Ground Truth: This transaction IS actually legitimate.")
