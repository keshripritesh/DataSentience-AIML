# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
import joblib
import os

def train_model(data_path='cleaned_data.csv', model_path='models/rf_model.pkl'):
    print("[📥] Loading cleaned dataset...")
    df = pd.read_csv(data_path)

    print("[🔖] Encoding target labels...")
    le = LabelEncoder()
    df['Risk_encoded'] = le.fit_transform(df['Risk'])

    # Split features and target
    X = df.drop(['Risk', 'Risk_encoded'], axis=1)
    y = df['Risk_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[🤖] Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Handle varying class presence in test set
    labels = unique_labels(y_test, y_pred)
    print("[📊] Classification Report:\n")
    print(classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=le.inverse_transform(labels)
    ))

    # Save model and label encoder
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((model, le), model_path)
    print(f"[✅] Model saved to: {model_path}")

if __name__ == "__main__":
    train_model()
