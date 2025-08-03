# predict.py

import pandas as pd
import joblib

def predict_spike(csv_path):
    df = pd.read_csv(csv_path)
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'Holidays_Count', 'Days', 'Month']
    latest = df[features].dropna().tail(1)

    scaler = joblib.load("model/feature_scaler.pkl")
    model = joblib.load("model/rf_spike_model.pkl")

    X_input = scaler.transform(latest)
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    return pred, prob

if __name__ == "__main__":
    result, confidence = predict_spike("data/final_dataset.csv")
    print(f"🚨 AQI Spike Tomorrow: {'YES' if result else 'NO'} (Confidence: {confidence:.2%})")
