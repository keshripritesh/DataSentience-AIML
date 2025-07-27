💰 Loan Approval Prediction System
This project predicts whether a loan application should be approved or rejected based on the applicant's financial and personal details, using a trained Machine Learning model.

[!ui screenshot](assets/1e.jpeg)
[!ui screenshot](assets/1f.jpeg)

✅ Features
Predicts loan approval status from user inputs

Streamlit-based UI for interactive usage

Random Forest Classifier for prediction

Command-Line Interface (CLI) via src/predict.py

Preprocessing pipeline with one-hot encoding and missing value imputation

Easily extendable and ready for deployment

🧠 Dataset
Source: Kaggle – Loan Prediction Dataset

File: data/loan_train.csv

Features:

Gender, Married, Dependents, Education, Self_Employed

ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term

Credit_History, Property_Area

Target: Loan_Status (1 = Approved, 0 = Rejected)