# predict.py

import pandas as pd
import numpy as np
from keras.models import load_model
import joblib

def load_recent_data(csv_path, past_days=7):
    df = pd.read_csv(csv_path)
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'Holidays_Count', 'Days', 'Month']
    recent_data = df[features].dropna().tail(past_days)

    feature_scaler = joblib.load("model/feature_scaler.pkl")
    target_scaler = joblib.load("model/target_scaler.pkl")

    scaled_recent = feature_scaler.transform(recent_data)
    return np.expand_dims(scaled_recent, axis=0), target_scaler

def predict_next_day_aqi():
    model = load_model("model/lstm_aqi_model.h5")
    X_input, target_scaler = load_recent_data("data/final_dataset.csv")
    y_pred_scaled = model.predict(X_input)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    return y_pred[0][0]

if __name__ == "__main__":
    prediction = predict_next_day_aqi()
    print(f"📈 Predicted AQI for next day: {prediction:.2f}")
