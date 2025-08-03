# predict.py
import pandas as pd
import joblib
from preprocess import preprocess_data

# Load model
model = joblib.load("model/rf_regressor.pkl")

# Sample input
sample_input = {
    "Confidence based on Introduction (English)": "Impactful - Good confidence",
    "Confidence based on the topic given": "Guarded Confidence",
    "Structured Thinking (In regional only)": "Guarded Confidence",
    "Regional fluency based on the topic given": "Taking gaps while speaking",
    "Mode of interview given by candidate?": "Mobile",
    "Experienced candidate - (Experience in months)": 12
}

# Convert to DataFrame
df = pd.DataFrame([sample_input])

# Preprocess (no target)
X_new, _ = preprocess_data(df, is_train=False)

# Predict
prediction = model.predict(X_new)
print("Predicted Total Score:", round(prediction[0], 2))
