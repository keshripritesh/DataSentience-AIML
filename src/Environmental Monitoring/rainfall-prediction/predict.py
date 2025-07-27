# predict.py

import pandas as pd
import joblib

def load_model(model_path='models/rf_model.pkl'):
    return joblib.load(model_path)

def predict_rain(input_data, model_path='models/rf_model.pkl'):
    model = load_model(model_path)
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return "Rain" if prediction == 1 else "No Rain"

if __name__ == "__main__":
    sample_input = {
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

    result = predict_rain(sample_input)
    print(f"[🔮] Prediction: {result}")
