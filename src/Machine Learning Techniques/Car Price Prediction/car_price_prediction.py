
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
# The dataset should be in the same directory as this script.
# You can download it from here: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
try:
    df = pd.read_csv('car data.csv')
except FileNotFoundError:
    print("Error: 'car data.csv' not found. Please download the dataset and place it in the same directory as the script.")
    exit()

# Preprocessing
df.drop('Car_Name', axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

# Feature Selection
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# 2. Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
y_pred_poly = poly_reg.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)

# 3. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

# 4. Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)


# --- Output ---
print("--- Car Price Prediction Model ---")
print(f"Linear Regression MSE: {mse_lr:.2f}")
print(f"Polynomial Regression MSE: {mse_poly:.2f}")
print(f"Ridge Regression MSE: {mse_ridge:.2f}")
print(f"Lasso Regression MSE: {mse_lasso:.2f}")

# --- Example Prediction ---
# You can create a new data point (as a DataFrame) to predict its price.
# Make sure it has the same columns as the training data.
new_car_data = {
    'Year': [2018],
    'Present_Price': [10.0],
    'Kms_Driven': [50000],
    'Owner': [0],
    'Fuel_Type_Diesel': [0],
    'Fuel_Type_Petrol': [1],
    'Selling_type_Individual': [0],
    'Transmission_Manual': [1]
}

new_car_df = pd.DataFrame(new_car_data)

# Ensure the new data has the same columns as the training data, adding missing columns with 0
for col in X_train.columns:
    if col not in new_car_df.columns:
        new_car_df[col] = 0

new_car_df = new_car_df[X_train.columns] # Ensure order is the same


# Predict with the best model (e.g., Linear Regression)
predicted_price = lr.predict(new_car_df)
print(f"\nPredicted price for the new car: {predicted_price[0]:.2f} lakhs")
