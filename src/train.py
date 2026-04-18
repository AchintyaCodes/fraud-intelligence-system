import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/creditcard.csv")

print("Dataset shape:", df.shape)
print(df.head())

# ----------------------------
# CHECK CLASS IMBALANCE
# ----------------------------
print("\nClass Distribution:")
print(df['Class'].value_counts())

sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud")
plt.show()

# ----------------------------
# FEATURE SCALING (IMPORTANT)
# ----------------------------
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Drop Time (not useful)
df = df.drop(['Time'], axis=1)

# ----------------------------
# SPLIT DATA
# ----------------------------
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train))
# ----------------------------
# BASELINE MODEL
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ----------------------------
# EVALUATION
# ----------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))