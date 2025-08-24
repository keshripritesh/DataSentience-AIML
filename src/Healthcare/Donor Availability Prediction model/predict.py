# predict.py
import argparse
import joblib
import pandas as pd

MODEL_PATH = "model/donor_availability.pkl"

def predict_availability(months_since_first_donation, number_of_donation, pints_donated, city, blood_group):
    # Load model & encoders
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    encoders = bundle["encoders"]

    # Prepare input
    data = pd.DataFrame([{
        "months_since_first_donation": months_since_first_donation,
        "number_of_donation": number_of_donation,
        "pints_donated": pints_donated,
        "city": encoders["city"].transform([city])[0],
        "blood_group": encoders["blood_group"].transform([blood_group])[0]
    }])

    # Predict
    pred = model.predict(data)[0]
    availability = encoders["availability"].inverse_transform([pred])[0]
    return availability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict donor availability")
    parser.add_argument("--months", type=int, required=True, help="Months since first donation")
    parser.add_argument("--donations", type=int, required=True, help="Number of donations")
    parser.add_argument("--pints", type=int, required=True, help="Pints donated")
    parser.add_argument("--city", type=str, required=True, help="City name")
    parser.add_argument("--blood", type=str, required=True, help="Blood group")

    args = parser.parse_args()

    result = predict_availability(
        args.months, args.donations, args.pints, args.city, args.blood
    )
    print(f"🩸 Predicted Availability: {result}")
