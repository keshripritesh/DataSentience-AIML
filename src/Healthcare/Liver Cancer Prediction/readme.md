# Synthetic Liver Cancer Dataset

This repository contains the **Synthetic Liver Cancer Dataset**, a medically realistic dataset designed for machine learning and healthcare research.  
It includes **5,000 patient records** with **14 features**, making it useful for binary classification tasks and reproducible ML projects.

---

## 📂 Dataset Overview

- **File:** `synthetic_liver_cancer_dataset.csv`  
- **Rows:** 5,000 patient records  
- **Features:** 14 clinical and lifestyle attributes  
- **Target:** Binary label (presence/absence of liver cancer)

### Example Features
- `age` – Patient age in years  
- `gender` – Male/Female  
- `bmi` – Body Mass Index  
- `smoking_status` – Never, Former, Current  
- `alcohol_consumption` – Low, Moderate, High  
- `family_history` – Yes/No  
- `lab_results` – Key biochemical measurements  

---

## 🚀 Proposed Contributions

### 1. Data Preprocessing Scripts
- Python utilities for handling:
  - Categorical encoding (`gender`, `smoking_status`, `alcohol_consumption`)  
  - Missing value imputation  
  - Feature scaling / normalization  

---

### 2. Exploratory Data Analysis (EDA)
- Interactive **Jupyter Notebook** including:
  - Histograms & boxplots for feature distributions  
  - Correlation heatmap for feature relationships  
  - Initial insights into target imbalance  

---

### 3. Baseline Machine Learning Models
- Ready-to-use classification models:
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Evaluation metrics:
  - Accuracy, Precision, Recall, F1-score  

---

### 4. Documentation Improvements
- Clear explanations of dataset columns and use cases  
- Example code snippets for loading and preparing the dataset  

```python
import pandas as pd

# Load dataset
df = pd.read_csv("synthetic_liver_cancer_dataset.csv")

# Train/test split
from sklearn.model_selection import train_test_split
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
