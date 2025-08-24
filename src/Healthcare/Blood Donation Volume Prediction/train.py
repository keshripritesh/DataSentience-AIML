# train.py
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import preprocess_data
from math import sqrt

MODEL_PATH = "model/donation_volume.pkl"

if __name__ == "__main__":
    # Preprocess
    X_train, X_test, y_train, y_test, encoders = preprocess_data()

    # Train model
    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(X_train, y_train)

    # Evaluate


# ...

    # Evaluate
    y_pred = reg.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))   # ✅ RMSE manually
    r2 = r2_score(y_test, y_pred)


    print("✅ Model trained")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Save model & encoders
    joblib.dump({"model": reg, "encoders": encoders}, MODEL_PATH)
    print(f"📁 Model saved at {MODEL_PATH}")
