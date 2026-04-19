🚀 Fraud Intelligence System
💳 AI-Powered Real-Time Credit Card Fraud Detection
<p align="center"> <img src="https://img.shields.io/badge/Machine%20Learning-XGBoost-blue?style=for-the-badge" /> <img src="https://img.shields.io/badge/Imbalance-SMOTE-orange?style=for-the-badge" /> <img src="https://img.shields.io/badge/Explainability-SHAP-purple?style=for-the-badge" /> <img src="https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge" /> <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge" /> </p> <p align="center"> <b>End-to-End ML System | Real-Time Predictions | Explainable AI</b> </p>
🌟 Live Demo

👉 (Add your Streamlit link here once deployed)

https://your-app.streamlit.app
🧠 What This Project Does

This is a production-style machine learning system that detects fraudulent credit card transactions in real-time using:

⚡ High-performance gradient boosting (XGBoost)
⚖️ Class imbalance handling (SMOTE)
🔍 Explainable AI (SHAP)
🌐 Interactive UI (Streamlit)
🎯 Key Highlights

✔ Real-time fraud prediction
✔ Batch fraud detection (CSV upload)
✔ Feature importance visualization (SHAP)
✔ Clean ML pipeline (no data leakage)
✔ Consistent preprocessing (scaler saved & reused)
✔ Deployable + scalable architecture

📊 Model Performance
<p align="center"> <img src="outputs/roc_curve.png" width="400"/> </p>
Metric	Score
🔥 ROC-AUC	0.968
🎯 Accuracy	99.85%
⚠ Fraud Recall	70.59%
📸 UI Preview
<p align="center"> <img src="outputs/class_distribution.png" width="400"/> <img src="outputs/shap_summary.png" width="400"/> </p>

⚡ Beautiful dark-themed dashboard with real-time analytics

⚙️ Architecture
🧩 Project Structure
fraud-intelligence-system/
│
├── dashboard/          # Streamlit UI
│   └── app.py
│
├── src/                # ML pipeline
│   ├── train.py
│   ├── preprocessing.py
│   └── model.py
│
├── models/             # Saved artifacts
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── outputs/            # Visualizations
│   ├── roc_curve.png
│   ├── shap_summary.png
│   └── class_distribution.png
│
├── data/               # Dataset
├── requirements.txt
└── README.md
🚀 Installation & Run
1️⃣ Clone Repository
git clone https://github.com/AchintyaCodes/fraud-intelligence-system.git
cd fraud-intelligence-system
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Train Model
python src/train.py

✔ Saves:

model
scaler
plots
4️⃣ Run App
streamlit run dashboard/app.py
🧪 How to Use
🔹 Real-Time Mode
Generate random transaction
Click Analyze
View fraud probability instantly
🔹 Batch Mode
Upload CSV
Run detection
Download results
🧠 Feature Explanation
Feature	Description
V1–V28	PCA-transformed confidential features
Amount	Transaction value (scaled)
Class	Target (0 = legit, 1 = fraud)
🔍 Explainability (SHAP)
<p align="center"> <img src="outputs/shap_summary.png" width="600"/> </p>

👉 Shows which features contribute most to fraud detection

🔥 Future Enhancements
🔌 REST API (FastAPI / Flask)
📡 Real-time streaming fraud detection
🤖 Deep learning models
📱 Mobile dashboard
🌍 Cloud deployment with custom domain
👨‍💻 Author

Achintya Gupta
🎓 Data Science & Engineering @ MIT Manipal
🚀 Future ML Engineer

⭐ Support

If you like this project:

👉 Drop a ⭐ on GitHub
👉 Share with others
