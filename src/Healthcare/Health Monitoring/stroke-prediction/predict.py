import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_data

# Load model and training columns
@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

clf, expected_cols = load_model()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🧠 Stroke Risk Predictor", layout="centered")
st.title("🧠 Stroke Risk Prediction App")
st.write("Enter the details below to check the risk of stroke.")

with st.form("stroke_form"):
    col1, col2 = st.columns(2)

    with col1:
        id = st.number_input("Patient ID", min_value=1, value=101)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 1, 120, 40)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])

    with col2:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level", value=95.6, step=0.1)
        bmi = st.number_input("BMI", value=24.3, step=0.1)
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    submitted = st.form_submit_button("🔍 Predict Stroke Risk")

    if submitted:
        # Prepare input as a DataFrame
        sample = {
            'id': [id],
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status],
            'stroke': [0]  # dummy value
        }

        df = pd.DataFrame(sample)

        # Preprocess input
        X, _ = preprocess_data(df)

        # Align with training columns
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_cols]

        # Predict
        prediction = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]

        st.subheader(f"🧠 Prediction for ID {id}:")
        st.success(f"Stroke Risk: **{'Yes' if prediction == 1 else 'No'}**")
        st.write(f"📊 Probability:")
        st.write(f"- No Stroke: `{proba[0]:.2f}`")
        st.write(f"- Stroke: `{proba[1]:.2f}`")
