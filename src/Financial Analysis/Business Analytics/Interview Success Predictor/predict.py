# predict.py
import pandas as pd
import joblib
from preprocess import preprocess_data

# Load model & encoders
model = joblib.load("model/decision_tree_model.pkl")
encoders = joblib.load("model/encoders.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

# Sample input data (as dict)
sample_input = {
    "Type of Graduation/Post Graduation": "B.E / B-Tech",
    "Mode of interview given by candidate?": "Mobile",
    "Gender": "Male",
    "Experienced candidate - (Experience in months)": 12,
    "Confidence Score": 11,
    "Structured Thinking Score": 7,
    "Regional Fluency Score": 6,
    "Total Score": 55
}

# Convert to DataFrame
df = pd.DataFrame([sample_input])

# Encode categorical features
for col, le in encoders.items():
    df[col] = le.transform(df[col].astype(str))

# Predict
prediction = model.predict(df)
predicted_label = target_encoder.inverse_transform(prediction)
print(f"Predicted Interview Verdict: {predicted_label[0]}")
