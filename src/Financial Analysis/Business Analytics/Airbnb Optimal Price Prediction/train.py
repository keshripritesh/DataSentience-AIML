# train.py

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import load_and_preprocess_data

# Paths
DATA_PATH = 'data/airbnb_nyc.csv'
MODEL_PATH = 'model/price_model.pkl'

def train():
    # Load and preprocess data
    X, y, preprocessor = load_and_preprocess_data(DATA_PATH)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build training pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
    r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))

    print(f"RMSE: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Save model
    os.makedirs('model', exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
