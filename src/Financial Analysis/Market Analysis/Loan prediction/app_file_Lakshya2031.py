# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 23:08:04 2025

@author: HP
"""

import pickle
import streamlit as st
loan_load=pickle.load(open("D:\loan\loan.pkl","rb"))
def main():
    st.title("Loan Prediction")
    Gender=st.number_input("1-> Male 0-> Female")	
    Married=st.number_input("1-> Married 0->Unmarried")
    Dependents=st.number_input("No of Dependents")
    Education=st.number_input("1-> Graduate 0->Non Graduate")	
    Self_Employed=st.number_input("1->Yes 0-> No")	
    ApplicantIncome=st.number_input("Income of Applicant")
    CoapplicantIncome=st.number_input("Coapplicant income")	
    LoanAmount=st.number_input("Enter the amount")
    Loan_Amount_Term=st.number_input("Enter the loan paying duration")
    Credit_History=st.number_input("Credit_History 1-> clear 0->Not clear")	
    Property_Area=st.number_input("0-> For rural 1->Semiurban 2->Urban")
    loan_p=''
    if st.button("Predict loan Approval"):
        loan_predict=loan_load.predict([[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]])
        if loan_predict==0:
            loan_p='Not Approved'
        else:
            loan_p='Approved'
            
    st.success(loan_p)
if __name__=='__main__':
    main()
        
