# 🚗 Used Car Price Predictor with Machine Learning

A machine learning-based regression model that predicts the **market price** of a used car based on its specifications like mileage, engine, transmission, location, and more.

---

[!ui screenshot](assets/image.png)

## 📌 Problem Statement

Used car pricing varies drastically based on region, features, and past usage. Buyers and sellers often struggle to evaluate a fair market price. This project uses machine learning to build a robust **price estimation model** based on actual car data.

---

## 📊 Dataset

The dataset used is `cleaned_car_data.csv`, derived from real-world car listings with noisy columns removed. It contains structured data like:

| Feature             | Description                                  |
|---------------------|----------------------------------------------|
| `myear`             | Manufacturing year                           |
| `fuel`              | Fuel type (Petrol/Diesel/CNG/etc.)           |
| `transmission`      | Manual/Automatic transmission                 |
| `km`                | Kilometers driven                            |
| `body`              | Body type (Sedan, Hatchback, SUV)            |
| `Color`             | Car color                                    |
| `Engine Type`       | Type of engine (DOHC, SOHC, etc.)            |
| `No of Cylinder`    | Engine cylinder count                        |
| `Length`, `Width`, `Height` | Dimensions in mm                     |
| `Seats`             | Seating capacity                             |
| `Gear Box`          | Type of gearbox                              |
| `Drive Type`        | Front/Rear/All Wheel Drive                   |
| `Steering Type`     | Power/Manual Steering                        |
| `owner_type`        | First owner, second, etc.                    |
| `state`, `City`     | Geographical location                        |
| `listed_price`      | 🔥 Target column – the price to predict      |

---

## 🧠 ML Approach

- **Model**: Random Forest Regressor
- **Preprocessing**:  
  - Imputation (median for numerical, most frequent for categorical)  
  - One-Hot Encoding for categorical columns  
- **Pipeline**: Combined preprocessor + regressor with `sklearn.Pipeline`


