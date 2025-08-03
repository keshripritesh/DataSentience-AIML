# predict.py

import joblib
import pandas as pd
import numpy as np

MODEL_PATH = 'model/price_model.pkl'

def predict_price(input_data):
    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([input_data])
    log_price = model.predict(input_df)[0]
    predicted_price = np.expm1(log_price)
    return round(predicted_price, 2)

if __name__ == "__main__":
    sample_input = {
        "neighbourhood_group": "Brooklyn",
        "room_type": "Private room",
        "latitude": 40.6782,
        "longitude": -73.9442,
        "minimum_nights": 2,
        "availability_365": 250,
        "number_of_reviews": 55,
        "reviews_per_month": 1.8,
        "calculated_host_listings_count": 1
    }

    price = predict_price(sample_input)
    print(f"Recommended Price: ${price}")
