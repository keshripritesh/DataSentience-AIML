# predict.py

import pandas as pd
import joblib

def load_model(model_path='models/rf_model.pkl'):
    return joblib.load(model_path)

def prepare_input(data_dict):
    # Map 'GENDER': M/F to 1/0
    data_dict['GENDER'] = 1 if data_dict['GENDER'] == 'M' else 0
    return pd.DataFrame([data_dict])

def predict_risk(input_data, model_path='models/rf_model.pkl'):
    model = load_model(model_path)
    input_df = prepare_input(input_data)

    # Predict
    prediction = model.predict(input_df)[0]
    result = "YES (At Risk)" if prediction == 1 else "NO (Not at Risk)"
    return result

if __name__ == "__main__":
    sample_input = {
        'GENDER': 'M',
        'AGE': 65,
        'SMOKING': 1,
        'YELLOW_FINGERS': 1,
        'ANXIETY': 1,
        'PEER_PRESSURE': 1,
        'CHRONIC_DISEASE': 1,
        'FATIGUE': 1,
        'ALLERGY': 0,
        'WHEEZING': 1,
        'ALCOHOL_CONSUMING': 1,
        'COUGHING': 1,
        'SHORTNESS_OF_BREATH': 1,
        'SWALLOWING_DIFFICULTY': 1,
        'CHEST_PAIN': 1
    }

    risk = predict_risk(sample_input)
    print(f"[🔮] Predicted Lung Cancer Risk: {risk}")
