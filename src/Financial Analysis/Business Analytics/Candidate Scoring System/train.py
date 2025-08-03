# train.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import preprocess_data

# Load data
df = pd.read_csv("data/Data - Base.csv")

# Preprocess
X, y = preprocess_data(df)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model/rf_regressor.pkl")

# Evaluate
y_pred = model.predict(X)
print("R² Score:", r2_score(y, y_pred))
print("RMSE:", mean_squared_error(y, y_pred, squared=False))
