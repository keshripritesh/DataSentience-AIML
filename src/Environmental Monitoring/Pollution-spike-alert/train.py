# train.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocess import load_and_preprocess
import joblib
import os

if __name__ == "__main__":
    X, y = load_and_preprocess("data/final_dataset.csv")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/rf_spike_model.pkl")

    y_pred = model.predict(X)
    print("📊 Training Classification Report:")
    print(classification_report(y, y_pred))
