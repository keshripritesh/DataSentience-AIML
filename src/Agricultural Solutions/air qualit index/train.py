import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from preprocess import preprocess_data

# Load dataset
df = pd.read_csv("data/air_quality.csv")  # Update filename as needed

# Preprocess for training
X, y = preprocess_data(df, is_train=True)

# Train the model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X, y)

# Save model
joblib.dump(model, "models/aqi_model.pkl")
print("✅ Model saved to models/aqi_model.pkl")

# Evaluate on training set (for reference)
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\n📊 Training Performance:")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")
