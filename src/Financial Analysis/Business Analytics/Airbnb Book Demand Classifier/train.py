# train.py

import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from preprocess import load_and_preprocess_data

# Paths
DATA_PATH = 'data/airbnb_nyc.csv'
MODEL_PATH = 'model/booking_model.pkl'
PREPROCESSOR_PATH = 'model/preprocessor.pkl'

def train_model():
    # Load and preprocess
    X, y, preprocessor = load_and_preprocess_data(DATA_PATH)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build full pipeline
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs('model', exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
