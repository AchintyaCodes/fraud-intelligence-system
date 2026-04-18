import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Fraud Intelligence System", layout="wide")

# ----------------------------
# CUSTOM UI (🔥 AESTHETIC)
# ----------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

[data-testid="stSidebar"] {
    background: #020617;
}

.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
}

h1, h2, h3 {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("models/fraud_model.pkl")

# ----------------------------
# SIDEBAR INPUT
# ----------------------------
st.sidebar.title("🛠 Transaction Inputs")

amount = st.sidebar.number_input("Transaction Amount", value=100.0)

if st.sidebar.button("🎲 Generate Random Transaction"):
    features = np.random.normal(0, 1, 28)
else:
    features = np.zeros(28)

# Combine input
input_data = np.array([amount] + list(features)).reshape(1, -1)

# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown("""
# 💳 Fraud Intelligence System  
### ⚡ AI-powered Real-Time Credit Card Fraud Detection
""")

st.markdown("---")

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("🔍 Analyze Transaction"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Fraud Probability", f"{prob:.2%}")

    with col2:
        st.metric("Risk Level", "🚨 HIGH" if prediction == 1 else "✅ LOW")

    st.markdown("---")

    # ----------------------------
    # MODEL INSIGHTS (IMAGES)
    # ----------------------------
    st.subheader("📊 Model Insights")

    if os.path.exists("outputs/class_distribution.png"):
        st.image("outputs/class_distribution.png", caption="Class Distribution")

    if os.path.exists("outputs/roc_curve.png"):
        st.image("outputs/roc_curve.png", caption="ROC Curve")

    if os.path.exists("outputs/shap_summary.png"):
        st.image("outputs/shap_summary.png", caption="SHAP Feature Importance")


# ----------------------------
# BATCH FRAUD DETECTION
# ----------------------------
import pandas as pd

st.markdown("---")
st.subheader("📂 Batch Fraud Detection")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    if st.button("🚀 Run Fraud Detection on File"):
        # Drop Class if exists
        if "Class" in df.columns:
            df = df.drop("Class", axis=1)

        # Scale Amount
        if "Amount" in df.columns:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df["Amount"] = scaler.fit_transform(df[["Amount"]])

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df["Fraud_Prediction"] = predictions
        df["Fraud_Probability"] = probabilities

        st.success("✅ Prediction completed!")

        st.write("Results:")
        st.dataframe(df.head())

        fraud_count = df["Fraud_Prediction"].sum()
        total = len(df)

        col1, col2 = st.columns(2)
        col1.metric("Total Transactions", total)
        col2.metric("Fraud Detected", int(fraud_count))

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Results CSV",
            data=csv,
            file_name="fraud_results.csv",
            mime="text/csv"
        )
# ----------------------------
# FOOTER
# ----------------------------
st.markdown("""
---
Built by Achintya Gupta🚀 | Data Science & Engineering | MIT Manipal
""")