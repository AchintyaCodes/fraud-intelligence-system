import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("models/fraud_model.pkl")

# ----------------------------
# HEADER
# ----------------------------
st.markdown("# 💳 AI Fraud Detection System")
st.markdown("### Detect fraudulent transactions using Machine Learning")

st.divider()

# ----------------------------
# INPUT SECTION
# ----------------------------
st.subheader("🔢 Enter Transaction Features")

col1, col2 = st.columns(2)

features = {}

for i in range(1, 15):
    features[f"V{i}"] = col1.number_input(f"V{i}", value=0.0)

for i in range(15, 29):
    features[f"V{i}"] = col2.number_input(f"V{i}", value=0.0)

amount = st.number_input("💰 Transaction Amount", value=0.0)

input_data = pd.DataFrame([features])
input_data["Amount"] = amount

st.divider()

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("🚀 Predict Fraud"):

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"🚨 Fraud Detected!\n\nConfidence: {prob:.4f}")
    else:
        st.success(f"✅ Legit Transaction\n\nConfidence: {prob:.4f}")

    st.progress(float(prob))

    st.divider()

    # ----------------------------
    # SHAP EXPLANATION
    # ----------------------------
    st.subheader("🔍 Model Explanation (SHAP)")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)