# 🪨 Rock Vs Mine Classification Using Machine Learning

This project is a binary classification task that aims to distinguish between sonar signals bounced off **mines** and **rocks** using supervised machine learning algorithms. It demonstrates a practical application of signal classification and includes preprocessing, model training, evaluation, and interpretation.

---

## 🧰 Project Overview

- **Goal**: Predict whether a sonar signal belongs to a mine or a rock.
- **Type**: Binary Classification
- **Tech Stack**: Python, Scikit-learn, NumPy, Pandas, Matplotlib
- **ML Model Used**: Logistic Regression
- **Environment**: Google Colab

---

## 🗂 Project Directory Structure

src/
└── Miscellaneous/
└── Sonar Classification/
├── Rock_Vs_Mine_Classification.ipynb
└── README.md


---

## 📊 Dataset Information

- **Source**: [UCI Machine Learning Repository – Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))
- **Instances**: 208
- **Attributes**: 60 numerical features representing sonar energy in various frequency bands
- **Target Classes**:
  - `R` = Rock
  - `M` = Mine

The sonar returns are captured at different angles and frequencies, and each instance is labeled accordingly.

---

## 🔬 Exploratory Data Analysis (EDA)

The notebook includes:
- Dataset preview
- Class distribution check
- Heatmap for feature correlation
- Visualization of class-wise feature means

  
---

## 🧠 Model Details

- **Algorithm**: Logistic Regression
- **Why?** Simple, interpretable, and effective for linear decision boundaries.
- **Preprocessing**:
  - Label encoding of target variable (`R` to 0, `M` to 1)
  - Train-Test split (80-20)
  - Standardization of features using `StandardScaler`

---

## 🧪 Evaluation Metrics

- **Accuracy**
- **Confusion Matrix**
- **Classification Report**
- **Manual Prediction Validation**

Example Output:

```text
Accuracy: 86.9%
[[23  3]
 [ 4 12]]


*📌 You can add images here later by uploading plots and linking:*

