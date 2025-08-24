import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data

MODEL_PATH = "model/profit_classifier.pkl"

if __name__ == "__main__":
    # Preprocess
    X_train, X_test, y_train, y_test, le_industry, le_hq, scaler = preprocess_data()

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model and encoders
    joblib.dump({
        "model": model,
        "le_industry": le_industry,
        "le_hq": le_hq,
        "scaler": scaler
    }, MODEL_PATH)

    print(f"✅ Model saved at {MODEL_PATH}")
