# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_model(data_path='cleaned_data.csv', model_path='models/rf_model.pkl'):
    df = pd.read_csv(data_path)

    X = df.drop(['rainfall'], axis=1)
    y = df['rainfall']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("[📊] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Rain", "Rain"]))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[✅] Model saved to: {model_path}")

if __name__ == "__main__":
    train_model()
