# Car Price Prediction Model

This project implements a machine learning model to predict the selling price of a car based on its attributes.

## Features

- **Linear Regression:** A simple linear model to establish a baseline.
- **Polynomial Regression:** To capture non-linear relationships between features.
- **Ridge and Lasso Regression:** Regularization techniques to prevent overfitting.

## Dataset

The model is trained on the "Car Dekho" dataset, which you can download from [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).

**Important:** Download the `car data.csv` file and place it in the same directory as the `car_price_prediction.py` script.

## How to Use

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the script:**
   ```bash
   python car_price_prediction.py
   ```

## Model Output

The script will output the Mean Squared Error (MSE) for each of the four models, giving you an idea of their performance. It will also provide a sample prediction for a new car.

## Extensions

This project can be extended by:

- **Adding more features:** Incorporate other car attributes like engine power, brand, etc.
- **Hyperparameter tuning:** Optimize the `alpha` parameter for Ridge and Lasso regression.
- **Trying other models:** Experiment with models like Gradient Boosting or Random Forest for potentially better accuracy.
