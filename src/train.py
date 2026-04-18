from preprocessing import preprocess
import pandas as pd
from model import get_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE
import joblib
import shap

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/creditcard.csv")
print("Original Dataset shape:", df.shape)

#  SAMPLE FOR SPEED
df = df.sample(50000, random_state=42)

print("Sampled Dataset shape:", df.shape)
print(df.head())

# ----------------------------
# CHECK CLASS IMBALANCE
# ----------------------------
print("\nClass Distribution:")
print(df['Class'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud")
plt.savefig("outputs/class_distribution.png")
plt.close()

# ----------------------------
# PREPROCESSING (FROM MODULE)
# ----------------------------
# Split BEFORE preprocessing
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply preprocessing separately
X_train = preprocess(X_train, save_scaler=True)
X_test = preprocess(X_test, load_scaler=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# SMOTE
# ----------------------------
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train))

# ----------------------------
# MODEL (FROM MODULE)
# ----------------------------
model = get_model()

print("\nStarting training...")
model.fit(X_train, y_train)
print("Model training done")

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(model, "models/fraud_model.pkl")
print("Model saved")

# ----------------------------
# PREDICTIONS
# ----------------------------
print("Making predictions...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Predictions done")

# ----------------------------
# EVALUATION
# ----------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", roc_auc)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("outputs/roc_curve.png")
plt.close()

# ----------------------------
# SHAP EXPLAINABILITY
# ----------------------------
print("\nGenerating SHAP explanations...")

explainer = shap.Explainer(model)
X_sample = X_test.sample(100, random_state=42)

shap_values = explainer(X_sample)

shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("outputs/shap_summary.png")
plt.close()

print("SHAP explanation saved")