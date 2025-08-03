# preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'Holidays_Count', 'Days', 'Month']
    target = 'AQI'

    df = df[features + [target]].dropna().reset_index(drop=True)

    # Create binary spike label (1 if AQI increases >20% next day)
    df['AQI_next'] = df[target].shift(-1)
    df.dropna(inplace=True)

    df['spike'] = (df['AQI_next'] - df['AQI']) / df['AQI'] > 0.2
    df['spike'] = df['spike'].astype(int)

    X = df[features]
    y = df['spike']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/feature_scaler.pkl")

    return X_scaled, y

if __name__ == "__main__":
    X, y = load_and_preprocess("data/final_dataset.csv")
    print(f"✅ Preprocessed data: X shape = {X.shape}, y distribution:\n{np.bincount(y)}")
