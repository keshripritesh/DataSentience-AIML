# 🧠 Interview Success Predictor

This project is a machine learning-based system that predicts the outcome of a candidate's interview using structured interview evaluation data. It helps recruiters and HR departments estimate whether a candidate will pass or fail the interview based on profile data and communication scores.

[!ui ss](assets/image.png)
---

## 📊 Dataset

The dataset used in this project (`Data - Base.csv`) contains information from over 21,000 interviews. It includes:

- Demographics (Age, Gender, Education)
- Interview scores (Confidence, Fluency, Structured Thinking)
- Interview modality (Mode of interview)
- Experience details
- Final interview verdict

**Target column**: `Interview Verdict`

Possible target values:
- `Reject`
- `Select`
- `Premium Select`
- `Borderline Select`
- `Borderline Reject`

---

## ✅ Features Used

From the original dataset, we selected the following features for training:

| Feature                                   | Description                             |
|-------------------------------------------|-----------------------------------------|
| Type of Graduation/Post Graduation        | Candidate's education background        |
| Mode of interview                         | Mobile or Laptop                        |
| Gender                                    | Male or Female                          |
| Experienced candidate - (Experience in months) | Total work experience                |
| Confidence Score                          | Numeric score from interviewer          |
| Structured Thinking Score                 | Numeric score from interviewer          |
| Regional Fluency Score                    | Numeric score from interviewer          |
| Total Score                               | Combined interview performance score    |

---

## 🤖 Model

We use a **Decision Tree Classifier** from scikit-learn to model the relationship between features and interview outcome.

Other models like SVM, RandomForest, and XGBoost can be explored for better accuracy in future iterations.

---

## 🗂 Folder Structure

interview_success_predictor/
├── data/
│ └── Data - Base.csv
├── model/
│ ├── decision_tree_model.pkl
│ ├── encoders.pkl
│ └── target_encoder.pkl
├── preprocess.py
├── train.py
├── predict.py
└── README.md

yaml
Copy
Edit

---

## ⚙️ Requirements

Make sure you have Python 3.7+ installed.

Install the dependencies:

```bash
pip install -r requirements.txt
requirements.txt

nginx
Copy
Edit
pandas
scikit-learn
joblib
🚀 How to Run
1. Train the Model
bash
Copy
Edit
python train.py
This will:

Load the dataset

Preprocess the data

Train a decision tree model

Save the model and encoders under model/

2. Make a Prediction
Edit the sample in predict.py with new candidate data:

python
Copy
Edit
sample_input = {
    "Type of Graduation/Post Graduation": "B.E / B-Tech",
    "Mode of interview given by candidate?": "Mobile",
    "Gender": "Male",
    "Experienced candidate - (Experience in months)": 12,
    "Confidence Score": 11,
    "Structured Thinking Score": 7,
    "Regional Fluency Score": 6,
    "Total Score": 55
}
Then run:

bash
Copy
Edit
python predict.py
Output:

sql
Copy
Edit
Predicted Interview Verdict: Select