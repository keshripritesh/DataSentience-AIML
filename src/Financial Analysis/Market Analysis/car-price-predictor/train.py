# train.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from preprocess import get_preprocessor

# Load data
df = pd.read_csv("data/cleaned_car_data.csv")

# Features & target
X = df.drop("listed_price", axis=1)
y = df["listed_price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
preprocessor = get_preprocessor()
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Train
model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/model.pkl")
print("✅ Model saved to models/model.pkl")
