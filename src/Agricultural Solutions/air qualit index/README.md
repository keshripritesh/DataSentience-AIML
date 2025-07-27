# 🌫️ AQI Prediction from Pollutant Data

This project predicts the **Air Quality Index (AQI)** based on key pollutant concentrations such as PM2.5, NO₂, SO₂, CO, etc., using a regression model trained on historical environmental data.

---

[!ui screenshot](assets/image.png)

## 📌 Problem Statement

Many rows in environmental datasets contain missing AQI values. This project builds a machine learning model to estimate missing AQI values using available pollutant data. It also enables AQI prediction for new, unseen samples.

---




---

## 📊 Dataset Overview

Columns in the dataset:
- **City**, **Date** – Location & timestamp
- **PM2.5**, **PM10**, **NO**, **NO2**, **NOx**, **NH3**, **CO**, **SO2**, **O3**
- **Benzene**, **Toluene**, **Xylene**
- **AQI**, **AQI_Bucket** (Target columns — often missing)

---

## 🧪 Model & Features

- **Model:** RandomForestRegressor
- **Features Used:**  
  `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`

- **Target:** AQI (Air Quality Index)

--
