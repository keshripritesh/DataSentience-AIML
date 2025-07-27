# 📊 Mutual Fund Risk Predictor

This project uses machine learning to predict the **risk category** (e.g., Low, High, Very High) of a mutual fund based on features like returns, NAV, AUM, and investment details.

---

[!ui screenshot](assets/image.png)

## 📁 Project Structure

```

mutual\_fund\_risk\_predictor/
├── data/
│   └── Mutual\_fund Data.csv         # Original dataset
├── cleaned\_data.csv                 # Preprocessed dataset (auto-generated)
├── preprocess.py                    # Cleans and encodes the data
├── train\_model.py                   # Trains the ML model
├── predict.py                       # Predicts risk for new mutual fund inputs
├── models/
│   └── rf\_model.pkl                 # Trained model + label encoder
└── README.md                        # Project overview

````

---

## 🚀 Getting Started

### ✅ 1. Install Requirements

```bash
pip install pandas scikit-learn joblib
````

### 🧹 2. Preprocess the Dataset

```bash
python preprocess.py
```

* Cleans columns like `%`, `Rs.`, and `cr`
* One-hot encodes categorical features
* Drops invalid rows
* Saves cleaned output to `cleaned_data.csv`

### 🤖 3. Train the Model

```bash
python train.py
```

* Trains a `RandomForestClassifier`
* Prints a classification report
* Saves model to `models/rf_model.pkl`

### 🔮 4. Predict on New Data

```bash
python predict.py
```

* Uses a sample mutual fund dictionary as input
* Predicts and prints the fund's risk category

---

## 🧠 Sample Input Format

```python
sample_input = {
    "AMC": "mahindra manulife mutual fund",
    "Fund Name": "Demo Fund",
    "Morning star rating": 3,
    "Value Research rating": 4,
    "1 month return": "5.50%",
    "NAV": 25.6,
    "1 Year return": "42.50%",
    "3 Year Return": "19.20%",
    "Minimum investment": "Rs.500.0",
    "Fund Manager": "Test Manager",
    "AUM": "123.45 cr",
    "Category": "Equity"
}
```

---

## 📈 Model Performance

Run `train.py` to see a detailed classification report (precision, recall, F1-score) for each risk category.

---
