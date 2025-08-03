# preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_and_preprocess(csv_path, past_days=7):
    df = pd.read_csv(csv_path)

    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'Holidays_Count', 'Days', 'Month']
    target = 'AQI'

    df = df[features + [target]].dropna().reset_index(drop=True)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(df[features])
    scaled_target = target_scaler.fit_transform(df[[target]])

    os.makedirs("model", exist_ok=True)
    joblib.dump(feature_scaler, 'model/feature_scaler.pkl')
    joblib.dump(target_scaler, 'model/target_scaler.pkl')

    # Create LSTM sequences
    X, y = [], []
    for i in range(past_days, len(scaled_features)):
        X.append(scaled_features[i - past_days:i])
        y.append(scaled_target[i])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_and_preprocess("data/final_dataset.csv")
    print(f"✅ Preprocessing complete: X shape = {X.shape}, y shape = {y.shape}")
