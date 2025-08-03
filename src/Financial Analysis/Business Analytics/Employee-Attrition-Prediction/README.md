# 🧠 Employee Attrition Prediction (HR AI)

This project predicts whether an employee is likely to **leave the organization** based on their personal and professional attributes. The model is trained on the **IBM HR Analytics Employee Attrition & Performance Dataset**, a popular dataset in the HR domain.

[!ui screenshot](assets/58.jpeg)
[!ui screenshot](assets/59.jpeg)
[!ui screenshot](assets/image.png)

this screenshot contains the predicted results the sample data contsins in the predict.py file
---

## 📊 Dataset

- **Source**: [Kaggle - IBM HR Analytics Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Target Column**: `Attrition` (`Yes` or `No`)
- **Features**: Age, Job Role, Marital Status, Overtime, Income, Travel Frequency, Job Satisfaction, and 30+ more HR-related attributes.

---

## 🧰 Tech Stack

- **Language**: Python
- **Model**: CatBoostClassifier (Gradient Boosted Decision Trees)
- **Libraries**:
  - `pandas`, `scikit-learn`, `catboost`
  - `joblib` (for model saving/loading)

---

## 🔍 Enhancements

- Added **manual upsampling** of the minority class to handle class imbalance.
- Engineered new features:
  - `EstimatedDailyHours` = DailyRate / HourlyRate
  - `EstimatedWeeklyHours` = EstimatedDailyHours × 5
  - `DeviationFromStandardHours` = EstimatedWeeklyHours - 40
- Accuracy improved from **88.4% to 97.9%** on the test set.


## 🛠️ Project Structure

