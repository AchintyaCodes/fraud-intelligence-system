import streamlit as st
import numpy as np
import joblib

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("models/fraud_model.pkl")

st.set_page_config(page_title="Fraud Intelligence System", layout="wide")

# -------------------------
# TITLE
# -------------------------
st.title("💳 Fraud Intelligence System")
st.markdown("### Real-time Credit Card Fraud Detection using ML")

# -------------------------
# INPUT SECTION
# -------------------------
st.sidebar.header("🔧 Transaction Inputs")

amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)

# Generate dummy PCA features (since dataset uses V1–V28)
features = [st.sidebar.slider(f"V{i}", -5.0, 5.0, 0.0) for i in range(1, 29)]

# Combine input
input_data = np.array(features + [amount]).reshape(1, -1)

# -------------------------
# PREDICT
# -------------------------
if st.sidebar.button("🚀 Predict Fraud"):

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED! (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Confidence: {1 - prob:.2f})")

# -------------------------
# VISUALS
# -------------------------
st.subheader("📊 Model Insights")

st.image("outputs/class_distribution.png", caption="Class Distribution")
st.image("outputs/roc_curve.png", caption="ROC Curve")
st.image("outputs/shap_summary.png", caption="Feature Importance (SHAP)")