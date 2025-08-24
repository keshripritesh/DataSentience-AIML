# train.py
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/stock_ai_generated.csv"
MODEL_PATH = "model/stock_impact_regressor.pkl"

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
print("Columns in dataset:", df.columns.tolist())

# Target column (you can adjust if your dataset differs)
target = "Stock_Impact_%"

X = df.drop(columns=[target])
y = df[target]

# 2. Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include=object)),
    ]
)

# 3. Define full pipeline with model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])
print("Columns in dataset:", df.columns.tolist())

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R²: {r2:.4f}")

# 7. Save model
joblib.dump(model, MODEL_PATH)
print(f"🎉 Model saved at {MODEL_PATH}")
