# train.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data

MODEL_PATH = "model/donor_availability.pkl"

if __name__ == "__main__":
    # Preprocess
    X_train, X_test, y_train, y_test, encoders = preprocess_data()

    # Train model
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("✅ Model trained")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model & encoders
    joblib.dump({"model": clf, "encoders": encoders}, MODEL_PATH)
    print(f"📁 Model saved at {MODEL_PATH}")
