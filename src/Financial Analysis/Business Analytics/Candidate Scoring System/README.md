📊 Candidate Scoring System (Regression)

This project uses machine learning to predict a candidate’s **Total Interview Score** based on their communication ability, thinking skills, and interview mode/experience. It is part of a larger analytics system that automates talent assessment based on structured HR data.
[!ui ss](assets/image.png)
---

## 🧠 Problem Statement

Companies often conduct structured interviews where evaluators give scores for:
- Confidence
- Structured thinking
- Regional fluency

But these are often **textual** ("Impactful", "Guarded") or inconsistent. This project converts these assessments into ordinal scores and uses them to train a machine learning model to predict the **Total Interview Score** — helping build a fast, data-driven candidate scoring system.

---

## 🗃️ Dataset Overview

> File: `Data - Base.csv`  
> Rows: ~21,000+  
> Columns: 50+ (structured + text)

The dataset includes:
- Candidate demographics
- Experience
- Mode of interview
- Textual evaluations of confidence, fluency, and thinking
- Final total interview score

---

## 🎯 Target Variable

- `Total Score` (continuous numeric value between ~0 and 100)

---

## 🔍 Features Used for Prediction

| Feature Name                                  | Description                              |
|----------------------------------------------|------------------------------------------|
| Confidence based on Introduction (English)   | Textual confidence rating                |
| Confidence based on the topic given          | Textual confidence rating                |
| Structured Thinking (In regional only)       | Logical reasoning rating (text)          |
| Regional fluency based on the topic given    | Communication fluency (text)             |
| Mode of interview given by candidate?        | Interview medium (Mobile/Laptop)         |
| Experienced candidate - (Experience in months)| Work experience in months (numeric)      |

Textual fields are converted to numeric ordinal scores:
- `Struggled` → 1  
- `Guarded` → 2  
- `Impactful` / `Good` → 3

---

## ⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib

---

## 🤖 Model

We use a **RandomForestRegressor**, a robust ensemble learning method suitable for tabular data, capable of modeling non-linear relationships.

> **R² Score**: ~0.817  
> **RMSE**: ~6.99

This indicates strong predictive performance on the training data.

---

## 📂 Folder Structure

candidate_scoring_system/
├── data/
│ └── Data - Base.csv
├── model/
│ └── rf_regressor.pkl
├── preprocess.py
├── train.py
├── predict.py
└── README.md

yaml
Copy
Edit

---

## 🛠 How to Run

### 1. Install Requirements

```bash
pip install pandas scikit-learn joblib
2. Train the Model
bash
Copy
Edit
python train.py
This will:

Load and clean the data

Convert text fields to numeric scores

Train a RandomForestRegressor

Save the model in model/rf_regressor.pkl

3. Predict for New Candidate
Edit predict.py with your custom input:

python
Copy
Edit
sample_input = {
    "Confidence based on Introduction (English)": "Impactful - Good confidence",
    "Confidence based on the topic given": "Guarded Confidence",
    "Structured Thinking (In regional only)": "Guarded Confidence",
    "Regional fluency based on the topic given": "Taking gaps while speaking",
    "Mode of interview given by candidate?": "Mobile",
    "Experienced candidate - (Experience in months)": 12
}
Then run:

bash
Copy
Edit
python predict.py
Example Output:

yaml
Copy
Edit
Predicted Total Score: 58.75
📊 Model Evaluation
Metric	Score
R² Score	0.817
RMSE	~6.99