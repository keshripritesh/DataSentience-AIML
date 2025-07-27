import pandas as pd
import joblib

# Load model
model = joblib.load("models/model.pkl")

# Sample input data
user_input = pd.DataFrame([{
    "age": 30,
    "sex": "female",
    "bmi": 26.5,
    "children": 1,
    "smoker": "no",
    "region": "southeast"
}])

# Predict
prediction = model.predict(user_input)[0]
print(f"💵 Predicted Insurance Premium: ${prediction:.2f}")
