import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_data

# Page config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

# Title
st.title("💰 Loan Approval Prediction System")
st.markdown("Fill in the applicant details below to check if the loan is likely to be approved.")

# Input form
with st.form("loan_form"):
    st.subheader("📋 Personal & Financial Information")

    col1, col2 = st.columns(2)
    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Married = st.selectbox("Married", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
        Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    with col2:
        ApplicantIncome = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0, value=2000.0)
        LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=100.0)
        Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0.0, value=360.0)
        Credit_History = st.selectbox("Credit History", [1.0, 0.0])

    submitted = st.form_submit_button("📤 Predict Loan Approval")

# On submit
if submitted:
    try:
        # Construct dataframe from inputs
        input_data = pd.DataFrame([{
            'Gender': Gender,
            'Married': Married,
            'Dependents': Dependents,
            'Education': Education,
            'Self_Employed': Self_Employed,
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': Property_Area
        }])

        st.info("🔄 Preprocessing input...")
        input_processed = preprocess_data(input_data, for_training=False)

        st.info("📦 Loading trained model...")
        model = joblib.load("models/model.pkl")

        st.info("🔍 Making prediction...")
        prediction = model.predict(input_processed)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
            st.balloons()
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
