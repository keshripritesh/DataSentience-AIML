import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load trained XGBoost model
model = load("xgb_model.pkl")

# Get feature names from model
feature_names = model.get_booster().feature_names
st.write("✅ Model expects these features:", feature_names)

st.title("🚀 NASA Asteroid Hazard Prediction")

# Build input fields dynamically for each feature
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(
        f"Enter {feature}", value=0.0, format="%.5f"
    )

# Convert to DataFrame with correct order
features = pd.DataFrame([input_data], columns=feature_names).astype(np.float32)

# Show entered data
st.write("📊 Input Features:", features)

# Predict button
if st.button("🔮 Predict"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("⚠ This asteroid is *Hazardous*")
    else:
        st.success("✅ This asteroid is *Not Hazardous*")