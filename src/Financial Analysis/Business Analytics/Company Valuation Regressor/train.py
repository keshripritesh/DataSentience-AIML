import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from preprocess import preprocess_data

MODEL_PATH = "model/company_valuation_model.pkl"

# Preprocess data
X_train, X_test, y_train, y_test, scaler, le_industry, le_hq = preprocess_data()

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & preprocessors
joblib.dump({
    "model": model,
    "scaler": scaler,
    "le_industry": le_industry,
    "le_hq": le_hq
}, MODEL_PATH)

print(f"Model saved at {MODEL_PATH}")
