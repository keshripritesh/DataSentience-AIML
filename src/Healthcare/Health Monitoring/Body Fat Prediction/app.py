import streamlit as st
import os
import joblib
import numpy as np

# Load model
with open('./model/linear_regression.lb', 'rb') as f:
    model = joblib.load(f)

# Streamlit UI
st.title("💪 Body Fat Percentage Predictor")
st.markdown("Provide the measurements below to estimate your body fat percentage.")

# Input fields
age = st.number_input("Age (years)", 10, 100, 25)
weight = st.number_input("Weight (lbs)", 50.0, 400.0, 160.0)
height = st.number_input("Height (inches)", 50.0, 90.0, 70.0)
neck = st.number_input("Neck (cm)", 10.0, 60.0, 37.0)
chest = st.number_input("Chest (cm)", 50.0, 150.0, 100.0)
abdomen = st.number_input("Abdomen (cm)", 50.0, 150.0, 90.0)
hip = st.number_input("Hip (cm)", 50.0, 150.0, 95.0)
thigh = st.number_input("Thigh (cm)", 30.0, 100.0, 55.0)
knee = st.number_input("Knee (cm)", 30.0, 80.0, 45.0)
ankle = st.number_input("Ankle (cm)", 15.0, 40.0, 23.0)
biceps = st.number_input("Biceps (cm)", 20.0, 60.0, 35.0)
forearm = st.number_input("Forearm (cm)", 15.0, 45.0, 28.0)
wrist = st.number_input("Wrist (cm)", 10.0, 25.0, 18.0)

# Predict button
if st.button("Predict Body Fat %"):
    input_data = np.array([[age, weight, height, neck, chest, abdomen, hip, thigh,
                            knee, ankle, biceps, forearm, wrist]])
    
    prediction = model.predict(input_data)[0]
    st.success(f"📉 Estimated Body Fat Percentage: **{prediction:.2f}%**")