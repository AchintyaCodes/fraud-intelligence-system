import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess(df, save_scaler=False):
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    if save_scaler:
        joblib.dump(scaler, "models/scaler.pkl")
        print("Scaler saved")

    df = df.drop(['Time'], axis=1)

    return df