import pandas as pd
import joblib
from preprocess import preprocess_data

# Load model
model = joblib.load("models/aqi_model.pkl")

# 🧪 Option 1: Predict from sample input
sample = pd.DataFrame([{
    "PM2.5": 95.0,
    "PM10": 120.0,
    "NO": 10.5,
    "NO2": 32.0,
    "NOx": 34.7,
    "NH3": 20.0,
    "CO": 1.4,
    "SO2": 22.0,
    "O3": 33.0,
    "Benzene": 0.4,
    "Toluene": 1.2,
    "Xylene": 0.05,
    "AQI": 0  # dummy (ignored)
}])

X, _ = preprocess_data(sample, is_train=False)
pred = model.predict(X)[0]
print(f"🔍 Predicted AQI for sample input: {pred:.2f}")

# 🧪 Option 2: Predict missing AQI values from full CSV
df_all = pd.read_csv("data/air_quality.csv")
missing_aqi = df_all[df_all["AQI"].isna()].copy()

if not missing_aqi.empty:
    X_missing, _ = preprocess_data(missing_aqi, is_train=False)
    preds = model.predict(X_missing)
    missing_aqi["Predicted_AQI"] = preds
    print("\n✅ Predicted AQI for missing entries:")
    print(missing_aqi[["Date", "City", "Predicted_AQI"]].head())
else:
    print("\n✅ No missing AQI values to predict.")
