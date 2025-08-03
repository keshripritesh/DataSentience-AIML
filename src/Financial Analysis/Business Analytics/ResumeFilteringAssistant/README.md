🧠 Resume Filtering Assistant

The **Resume Filtering Assistant** is a machine learning-based tool designed to assist HR teams and recruiters in shortlisting job candidates based on structured interview data. It predicts whether a candidate is likely to **join the company or not**, based on behavioral and performance indicators gathered during the interview process.

---

## 📌 Table of Contents

- [📍 Overview](#-overview)
- [📊 Dataset](#-dataset)
- [🧼 Preprocessing](#-preprocessing)
- [🤖 Model Training](#-model-training)
- [🧪 Inference](#-inference)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [📦 Dependencies](#-dependencies)
- [📌 Future Improvements](#-future-improvements)
- [🤝 Contribution](#-contribution)
- [📄 License](#-license)

---

## 📍 Overview

Recruiters often struggle to filter through large volumes of candidate interview data. This assistant helps automate that process by:

- Cleaning and encoding raw candidate data.
- Training a classifier to predict the likelihood of a candidate joining.
- Providing predictions on new candidate profiles.

> 🔍 Built with `Python`, `pandas`, `scikit-learn`, and `pickle`.

---

## 📊 Dataset

The dataset contains detailed structured information about each candidate, such as:

- Demographics: Age, Gender, Marital Status, etc.
- Education: Type of Graduation/Post Graduation.
- Interview performance: Confidence, fluency, structured thinking (in English & regional).
- HR feedback: Role acceptance, verdict, CTC, current employment, red flags, etc.
- Target label: `Whether joined the company or not` (Yes/No)

> ✅ This is a classification problem with a binary outcome.

---

## 🧼 Preprocessing

`preprocess.py` handles:

- Column trimming and duplicate removal.
- Dropping irrelevant fields like internal comments.
- Label encoding and one-hot encoding for categorical columns.
- Separation and reintegration of the target column (`Whether joined the company or not`).

---

## 🤖 Model Training

`train.py` performs the following:

1. Loads and preprocesses the dataset.
2. Splits it into training and test sets.
3. Scales the features using `StandardScaler`.
4. Trains a `RandomForestClassifier`.
5. Saves:
   - Trained model (`candidate_model.pkl`)
   - Scaler (`scaler.pkl`)
   - Training column list (`train_columns.pkl`)
6. Outputs a `classification_report` with accuracy, precision, recall, and F1-score.

---

## 🧪 Inference

`predict.py` supports two modes:

### 🔹 1. Predict from Sample

The script includes a sample dictionary of candidate details to simulate predictions.

```bash
python predict.py
It returns:

text
Copy
Edit
        Name Prediction
0  Shahan      Yes
🔹 2. Predict from New CSV
You can also modify predict_from_dataframe() to accept a new .csv file of unseen candidate data.

📁 Project Structure
kotlin
Copy
Edit
ResumeFilteringAssistant/
├── data/
│   └── interview_data.csv
├── models/
│   ├── candidate_model.pkl
│   ├── scaler.pkl
│   └── train_columns.pkl
├── preprocess.py
├── train.py
├── predict.py
├── README.md