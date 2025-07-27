# predict.py
import pandas as pd
import joblib

def predict_price(input_data):
    model = joblib.load("models/model.pkl")
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return round(prediction[0], 2)

if __name__ == "__main__":
    # Sample input — replace values
    sample = {
        'myear': 2018,
        'fuel': 'Petrol',
        'transmission': 'Manual',
        'km': 45000,
        'body': 'Hatchback',
        'Color': 'White',
        'Engine Type': 'DOHC',
        'No of Cylinder': 4,
        'Length': 3700,
        'Width': 1600,
        'Height': 1500,
        'Seats': 5,
        'Gear Box': '5 Speed',
        'Drive Type': 'FWD',
        'Steering Type': 'Power',
        'owner_type': 'First',
        'state': 'Maharashtra',
        'City': 'Mumbai'
    }

    result = predict_price(sample)
    print(f"💰 Predicted Car Price: ₹{result}")
