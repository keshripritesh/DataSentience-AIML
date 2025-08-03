# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("data/employee_attrition.csv")

# Drop non-useful columns
df.drop(['EmployeeNumber', 'Over18', 'EmployeeCount', 'StandardHours'], axis=1, inplace=True)

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categoricals
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].apply(lambda col: col.astype('category').cat.codes)

# Daily, weekly and DeviationFromStandardHours features added for readability

df["EstimatedDailyHours"] = df["DailyRate"] / df["HourlyRate"]
df["EstimatedWeeklyHours"] = df["EstimatedDailyHours"] * 5
standard_hours = 40
df["DeviationFromStandardHours"] = df["EstimatedWeeklyHours"]-standard_hours


# Handle missing values (if any)
df.fillna(0, inplace=True)

#add weekly work hours feature 


# Split features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

#Manual upscaling of minority class to boost accuracy

from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df[df["Attrition"] == 0]
df_minority = df[df["Attrition"] == 1]

# Upsample minority class to match majority count
df_minority_upsampled = resample(
    df_minority,
    replace=True,                     # sample with replacement
    n_samples=len(df_majority),       # match majority class count
    random_state=42
)

# Combine majority and upsampled minority
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Shuffle the rows
df_upsampled = df_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Update X and y
X = df_upsampled.drop("Attrition", axis=1)
y = df_upsampled["Attrition"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CatBoost model
model = CatBoostClassifier(verbose=0)
model.fit(X_train, y_train)

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/catboost_model.pkl")

print("✅ Model trained and saved to models/catboost_model.pkl")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {accuracy:.4f}")

# Before Upscaling the accuracy was 88.44%
