# train.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from preprocess import preprocess_data

# Load dataset
df = pd.read_csv("data/genai_tool_data.csv")

# Preprocess
X, y = preprocess_data(df)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model/genai_model.pkl")

# Evaluate
y_pred = model.predict(X)
print(classification_report(y, y_pred))
