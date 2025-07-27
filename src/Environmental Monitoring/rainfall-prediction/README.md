# 🌧️ Rainfall Predictor using ML

A machine learning model that predicts whether it will rain on a given day based on weather attributes like temperature, humidity, cloud cover, and wind.



[!ui screenshot](assets/image.png)
---

## 📂 Folder Structure

rainfall_predictor/
├── data/Rainfall.csv
├── cleaned_data.csv
├── preprocess.py
├── train_model.py
├── predict.py
├── models/rf_model.pkl
└── README.md

yaml
Copy
Edit

---

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install pandas scikit-learn joblib
2. Clean the Dataset
bash
Copy
Edit
python preprocess.py
3. Train the Model
bash
Copy
Edit
python train_model.py
4. Predict Rainfall
bash
Copy
Edit
python predict.py
🧪 Sample Input
python
Copy
Edit
{
  'day': 15,
  'pressure': 1018.6,
  'maxtemp': 21.5,
  'temparature': 20.0,
  'mintemp': 19.2,
  'dewpoint': 18.5,
  'humidity': 88,
  'cloud': 85,
  'sunshine': 0.4,
  'winddirection': 70.0,
  'windspeed': 16.7
}
📈 Output
Prediction: Rain or No Rain

🧠 Model
Random Forest Classifier

Accuracy depends on available features and seasonality

Classification report printed after training

