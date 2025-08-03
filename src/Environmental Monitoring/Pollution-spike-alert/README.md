🚨 Pollution Spike Warning System (AQI Alert)

This project builds a real-world **Air Quality Index (AQI) spike predictor** to forecast whether AQI will rise sharply (by more than 20%) the next day based on current pollution levels and date features. It uses a machine learning classification model (Random Forest) trained on time-series AQI data.

[!ui ss](assets/image.png)
---

## 🧠 Problem Statement

In many regions, sudden increases in AQI pose a risk to public health, traffic, and policy response. This system predicts whether **tomorrow’s AQI will spike significantly (>20%)** using today’s:

- Pollution indicators (e.g., PM2.5, PM10, NO2)
- Calendar features (e.g., day of the month, holidays)

The result is a **binary alert system** for early warnings and preventive action.

---

## 📁 Project Structure

pollution-spike-alert/
├── data/
│ └── final_dataset.csv # Input dataset with AQI and pollution features
├── model/
│ ├── rf_spike_model.pkl # Trained Random Forest classifier
│ └── feature_scaler.pkl # Scaler for input features
├── preprocess.py # Preprocesses data, creates binary labels, scales features
├── train.py # Trains the Random Forest model
├── predict.py # Predicts whether AQI will spike tomorrow
├── README.md


---

## 🧪 Data Description

Your dataset should be placed at:  
`data/final_dataset.csv`

Expected columns (minimum):

- **Pollution features**: `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `Ozone`
- **Date features**: `Holidays_Count`, `Days`, `Month`
- **Target feature**: `AQI`

---

## 🧾 Spike Labeling Logic

We define a **spike** as a situation where tomorrow’s AQI increases by more than 20% compared to today:

spike = ((AQI_tomorrow - AQI_today) / AQI_today) > 0.2


This creates a binary target column `spike`:
- `1`: AQI spike predicted
- `0`: AQI stable or decreasing

---

## ⚙️ Setup Instructions

### 1. 🔽 Install Dependencies


pandas
numpy
scikit-learn
joblib
2. 🔧 Preprocess and Train Model

python train.py
Reads the dataset

Scales the features using MinMaxScaler

Trains a RandomForestClassifier

Saves the trained model and scaler into model/

3. 🔮 Make Prediction

python predict.py
Output:


Edit
🚨 AQI Spike Tomorrow: YES (Confidence: 84.23%)