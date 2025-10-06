# 🌍 GDP Prediction Model  

This project predicts a Country's **Per Capita GDP** based on multiple socio-economic factors using machine learning.  

## 📊 Dataset  
Dataset used: [GDP Prediction Dataset (Kaggle)](https://www.kaggle.com/rutikbhoyar/gdp-prediction-dataset)  

## ⚙️ Models Tested  
Four regression models were trained and compared:  
- Linear Regression  
- Support Vector Regressor (SVM)  
- Random Forest Regressor  
- Gradient Boosting Regressor  

**Performance Ranking:**  
Random Forest > Gradient Boosting > Linear Regression > SVM  

## 🏆 Best Model Performance (Random Forest)  
Using all features in the dataset:  
- MAE: 2125.24  
- RMSE: 3051.71  
- R² Score: 0.8873  

## 💻 Streamlit App  
A Streamlit web app is included for interactive predictions.  
- Users can input feature values (like Population, Literacy Rate, etc.)  
- The trained Random Forest model predicts GDP in real time.  

