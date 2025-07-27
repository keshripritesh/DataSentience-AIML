import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/catboost_model.pkl")

model = load_model()

def predict_attrition(input_data: dict):
    input_df = pd.DataFrame([input_data]).astype("float32")
    prediction = model.predict(input_df)[0]
    return "Yes" if prediction == 1 else "No"

# ---------------- UI ----------------
st.set_page_config(page_title="Attrition Predictor", layout="wide")
st.title("💼 Employee Attrition Prediction")

with st.form("attrition_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: ["Female", "Male"][x])
        education = st.slider("Education", 1, 5, 3)
        business_travel = st.selectbox("Business Travel", [0, 1, 2], format_func=lambda x: ["Rarely", "Frequently", "Non-Travel"][x])
        department = st.selectbox("Department", [0, 1, 2], format_func=lambda x: ["HR", "R&D", "Sales"][x])
        education_field = st.selectbox("Education Field", [0, 1, 2, 3, 4, 5], format_func=lambda x: ["HR", "Life Sciences", "Marketing", "Medical", "Technical", "Other"][x])

    with col2:
        daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
        hourly_rate = st.number_input("Hourly Rate", 30, 100, 60)
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
        monthly_rate = st.number_input("Monthly Rate", 1000, 50000, 20000)
        percent_salary_hike = st.slider("Salary Hike (%)", 10, 25, 12)
        stock_option_level = st.slider("Stock Option Level", 0, 3, 1)

    with col3:
        overtime = st.selectbox("OverTime", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        performance_rating = st.slider("Performance Rating", 1, 4, 3)
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)

    with st.expander("📊 Advanced Attributes"):
        c1, c2, c3 = st.columns(3)
        with c1:
            distance_from_home = st.slider("Distance From Home (km)", 1, 50, 10)
            num_companies_worked = st.slider("Companies Worked", 0, 10, 3)
            total_working_years = st.slider("Total Working Years", 0, 40, 10)
        with c2:
            training_times_last_year = st.slider("Trainings Last Year", 0, 6, 3)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
            years_since_last_promotion = st.slider("Years Since Promotion", 0, 15, 1)
        with c3:
            job_involvement = st.slider("Job Involvement", 1, 4, 3)
            job_level = st.slider("Job Level", 1, 5, 2)
            job_role = st.selectbox("Job Role", range(9))  # Add proper mapping later
            marital_status = st.selectbox("Marital Status", [0, 1, 2], format_func=lambda x: ["Single", "Married", "Divorced"][x])
            years_in_current_role = st.slider("Years in Current Role", 0, 20, 3)
            years_with_curr_manager = st.slider("Years with Manager", 0, 17, 4)

    submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        user_input = {
            'Age': age,
            'BusinessTravel': business_travel,
            'DailyRate': daily_rate,
            'Department': department,
            'DistanceFromHome': distance_from_home,
            'Education': education,
            'EducationField': education_field,
            'EnvironmentSatisfaction': environment_satisfaction,
            'Gender': gender,
            'HourlyRate': hourly_rate,
            'JobInvolvement': job_involvement,
            'JobLevel': job_level,
            'JobRole': job_role,
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': marital_status,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies_worked,
            'OverTime': overtime,
            'PercentSalaryHike': percent_salary_hike,
            'PerformanceRating': performance_rating,
            'RelationshipSatisfaction': relationship_satisfaction,
            'StockOptionLevel': stock_option_level,
            'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_times_last_year,
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_current_role,
            'YearsSinceLastPromotion': years_since_last_promotion,
            'YearsWithCurrManager': years_with_curr_manager
        }

        result = predict_attrition(user_input)
        st.success(f"🧠 Prediction: **Attrition = {result}**")
