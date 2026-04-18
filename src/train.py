import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/creditcard.csv")

print("Original Dataset shape:", df.shape)

# 🔥 SAMPLE FOR SPEED (VERY IMPORTANT)
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

# Save instead of show
plt.savefig("outputs/class_distribution.png")
plt.close()

# ----------------------------
# FEATURE SCALING
# ----------------------------
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Drop Time
df = df.drop(['Time'], axis=1)

# ----------------------------
# SPLIT DATA
# ----------------------------
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# HANDLE IMBALANCE (SMOTE)
# ----------------------------
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train))

# ----------------------------
# MODEL TRAINING
# ----------------------------
# ----------------------------
# MODEL TRAINING
# ----------------------------
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=1,
    n_jobs=-1,
    random_state=42,
    eval_metric="logloss"
)

print("\nStarting training...")
model.fit(X_train, y_train)
print("Model training done")

# ----------------------------
# PREDICTIONS
# ----------------------------
print("Making predictions...")
y_pred = model.predict(X_test)
print("Predictions done")

# ----------------------------
# EVALUATION
# ----------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))