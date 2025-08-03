AQI Predictor using LSTM
This project predicts the next day's Air Quality Index (AQI) using a deep learning model (LSTM) trained on historical pollution data. It leverages past readings of PM2.5, PM10, NO2, SO2, CO, Ozone, and date-based features to forecast future air quality.

[!ui](assets/image.png)

📂 Folder Structure
graphql
Copy
Edit
AQI Predictor lstm/
├── data/
│   └── final_dataset.csv           # Raw input dataset
├── model/
│   ├── lstm_aqi_model.h5           # Saved LSTM model
│   ├── feature_scaler.pkl          # Scaler for input features
│   └── target_scaler.pkl           # Scaler for AQI output
├── preprocess.py                   # Data preprocessing & sequence creation
├── train.py                        # Train LSTM model
├── predict.py                      # Predict AQI for next day
├── README.md
📊 Dataset Info
Input Features:

PM2.5, PM10, NO2, SO2, CO, Ozone

Holidays_Count, Days, Month

Target:

AQI (Air Quality Index)

You must place the dataset in: data/final_dataset.csv.

🚀 How It Works
🔧 1. Preprocessing (preprocess.py)
Normalizes features and AQI

Creates sliding window sequences (7 past days → 1 target)

Saves scalers to disk

🧠 2. Model Training (train.py)
Trains an LSTM model with dropout

Saves the trained .h5 model and scalers

📈 3. Prediction (predict.py)
Loads recent 7-day data

Uses saved model to predict the next day’s AQI

Returns the inverse-scaled (real) AQI value

1. 🔽 Install Dependencies
tensorflow
scikit-learn
pandas
numpy
joblib

2. 🧪 Train the Model
python train.py

3. 🔮 Make a Prediction
python predict.py
Example Output:

📈 Predicted AQI for next day: 167.42