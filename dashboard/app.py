import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("models/fraud_model.pkl")

st.title("💳 Fraud Detection System")

st.write("Enter transaction details:")

# ----------------------------
# INPUT FEATURES
# ----------------------------
features = {}

for i in range(1, 29):
    features[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

amount = st.number_input("Amount", value=0.0)

# Convert to dataframe
input_data = pd.DataFrame([features])
input_data["Amount"] = amount

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict"):

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Detected! Probability: {prob:.4f}")
    else:
        st.success(f"✅ Legit Transaction. Probability: {prob:.4f}")

    # ----------------------------
    # SHAP EXPLANATION
    # ----------------------------
    st.subheader("🔍 Model Explanation (SHAP)")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)