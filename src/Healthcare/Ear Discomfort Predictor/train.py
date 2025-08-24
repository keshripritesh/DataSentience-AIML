import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data

MODEL_PATH = "model/ear_discomfort_predictor.pkl"

def train():
    X_train, X_test, y_train, y_test, encoders, target_encoder = preprocess_data()

    # Train model
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Only include labels that appear in y_test
    labels = sorted(list(set(y_test)))  
    target_names = target_encoder.inverse_transform(labels)

    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))

    # Save model + encoders
    joblib.dump(
        {"model": model, "encoders": encoders, "target_encoder": target_encoder},
        MODEL_PATH
    )
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
