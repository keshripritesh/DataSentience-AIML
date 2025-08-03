# 📱 Screen Time Exceedance Classifier

This project is a machine learning classifier that predicts whether a child is likely to exceed the recommended daily screen time limit. It is built using demographic and behavioral features such as age, gender, type of device used, and screen time usage patterns.

[!ui screenshot](assets/image.png)
---

## 🧠 Problem Statement

In the digital age, children are spending more time on screens than ever before. Prolonged screen time can lead to health issues like eye strain, anxiety, poor sleep, and decreased attention spans. 

**Goal:**  
Develop a binary classifier that predicts whether a child is exceeding the safe screen time threshold, helping parents, educators, and policymakers make informed decisions.

---

## 🗂️ Folder Structure

Screen-Time-Exceedance-Classifier/
├── assets/ # Optional: images, visualizations, etc.
├── data/ # Dataset CSV goes here
├── model/ # Trained ML model and encoders
├── predict.py # Script for making predictions
├── preprocess.py # Data preprocessing logic
├── train.py # Training pipeline
└── README.md # Project documentation

---

## 📌 Dataset Overview

- **Rows:** 9712
- **Columns:** 8
- **Target Variable:** `Exceeded_Recommended_Limit` (True/False)

### Features Used
| Feature                              | Type      | Description                                      |
|--------------------------------------|-----------|--------------------------------------------------|
| `Age`                                | Numeric   | Age of the child (8–18 years)                   |
| `Gender`                             | Categorical | Male / Female                                 |
| `Primary_Device`                     | Categorical | Smartphone, Laptop, TV, Tablet                 |
| `Urban_or_Rural`                     | Categorical | Living region of the child                    |
| `Educational_to_Recreational_Ratio`  | Float     | Ratio of educational to recreational screen use |

---

## 🔍 Target Definition

| Target Label                 | Description                                     |
|-----------------------------|-------------------------------------------------|
| `True`                      | Child exceeds the recommended screen time       |
| `False`                     | Child stays within recommended screen time limit |

---

## ⚙️ Model Details

- **Type:** Binary Classification  
- **Algorithm:** Random Forest Classifier  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score

---

## 🛠️ How to Use

### 1. 📥 Clone the Repository & Add Dataset

Put the dataset `Indian_Kids_Screen_Time.csv` inside the `data/` folder.

### 2. 🧹 Preprocessing & Training

python train.py
Encodes categorical features

Splits training and testing sets

Trains a Random Forest model

Saves the model and encoders in model/

3. 🤖 Run Predictions
Edit predict.py or call it directly:
python predict.py
Example code inside predict.py:

result = predict_exceedance(13, "Male", "Smartphone", "Urban", 0.42)
print("Exceeded recommended screen time?" , result)