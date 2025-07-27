# 💵 Health Insurance Premium Predictor

This project predicts medical insurance charges based on user attributes such as age, BMI, smoking status, and region using a machine learning regression model.

---

## 📊 Dataset
- **Source:** Kaggle - Medical Cost Personal Dataset
- **File:** `data/insurance.csv`

[!ui screenshot](assets/image.png)

### Features:
- `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Target:** `charges` (insurance premium)

---

## 🧠 Model
- **Model:** Random Forest Regressor
- **Pipeline:** OneHotEncoder for categoricals + model
- **File saved at:** `models/model.pkl`

---

