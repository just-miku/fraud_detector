# Fraud Detection AI System

## Overview

This project demonstrates a **Random Forest-based fraud detection system** using anonymized credit card transactions.
It predicts the probability of fraud for each transaction and outputs a **risk score** in real-time via a Streamlit dashboard.

The system is designed to simulate a **bank-grade fraud detection pipeline** for demonstration and interview purposes.

---

## Features

* **Random Forest classifier** with class-weight balancing
* **Real-time predictions** through a Streamlit web interface
* **Adjustable fraud risk threshold** for operational control
* **Demo buttons** for random legitimate and fraudulent transactions
* **Ground truth display** to verify prediction accuracy

---

## Folder Structure

```
fraud_detection_project/
│
├── creditcard.csv           # Dataset (public anonymized transactions)
├── train_model.py           # Training script
├── fraud_model.pkl          # Trained Random Forest model
├── scaler.pkl               # Feature scaler
├── app.py                   # Streamlit demo application
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the demo:

```bash
streamlit run app.py
```

---

## Model Training

The model is trained using **`train_model.py`**:

* Features are scaled using `StandardScaler`
* Random Forest classifier with **200 trees** and `class_weight='balanced'`
* Handles **class imbalance** to prioritize catching fraudulent transactions
* Evaluation metrics include:

  * Precision
  * Recall
  * F1-score
  * ROC-AUC
* The trained model and scaler are saved as:

  * `fraud_model.pkl`
  * `scaler.pkl`

---

## Demo / Usage

* **Load Random Legit Transaction** – Shows a random non-fraudulent transaction and prediction
* **Load Random Fraud Transaction** – Shows a random fraudulent transaction and prediction
* **Fraud Risk Threshold Slider** – Adjust sensitivity; lower threshold detects more frauds at the cost of some false positives

The demo ensures realistic outputs for interview and presentation purposes.

---

## Interview Talking Points

* Shows handling of **extremely imbalanced datasets**
* Demonstrates **decision threshold tuning** for operational risk control
* Explains **trade-offs between precision and recall**
* Uses **realistic transaction sampling** for demonstration
* Can be extended to production-ready pipelines with:

  * Batch predictions
  * Feature importance visualization
  * Confusion matrix and model evaluation charts

---

## License / Disclaimer

* Dataset is publicly available and anonymized
* This project is for **educational and demonstration purposes only**
* Not intended for actual banking transactions or live fraud detection
