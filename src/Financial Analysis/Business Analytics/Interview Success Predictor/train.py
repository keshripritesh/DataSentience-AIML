# train.py
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from preprocess import preprocess_data

# Load data
df = pd.read_csv("data/Data - Base.csv")

# Preprocess
X, y, encoders, target_encoder = preprocess_data(df)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save model & encoders
joblib.dump(model, "model/decision_tree_model.pkl")
joblib.dump(encoders, "model/encoders.pkl")
joblib.dump(target_encoder, "model/target_encoder.pkl")

# Evaluate
y_pred = model.predict(X)
print("Classification Report:")
print(classification_report(y, y_pred, target_names=target_encoder.classes_))
