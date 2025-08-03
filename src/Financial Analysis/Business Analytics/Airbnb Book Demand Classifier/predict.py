# predict.py

import joblib
import pandas as pd

MODEL_PATH = 'model/booking_model.pkl'

def predict_booking(input_data):
    # Load model
    clf = joblib.load(MODEL_PATH)

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict
    prediction = clf.predict(input_df)[0]
    probability = clf.predict_proba(input_df)[0][1]  # probability of class 1

    result = {
        "prediction": "Frequently Booked" if prediction == 1 else "Infrequently Booked",
        "confidence": round(probability, 2)
    }
    return result

# Example usage
if __name__ == "__main__":
    sample_input = {
        "neighbourhood_group": "Manhattan",
        "room_type": "Entire home/apt",
        "price": 150,
        "minimum_nights": 3,
        "number_of_reviews": 50,
        "availability_365": 200,
        "calculated_host_listings_count": 2
    }

    result = predict_booking(sample_input)
    print(result)
