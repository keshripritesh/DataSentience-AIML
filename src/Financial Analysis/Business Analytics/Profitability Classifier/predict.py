import joblib
import numpy as np

MODEL_PATH = "model/profit_classifier.pkl"

def predict_profitability(sales, assets, market_value, industry, headquarters):
    # Load model and preprocessors
    saved_objects = joblib.load(MODEL_PATH)
    model = saved_objects["model"]
    le_industry = saved_objects["le_industry"]
    le_hq = saved_objects["le_hq"]
    scaler = saved_objects["scaler"]

    # Transform categorical features
    industry_enc = le_industry.transform([industry])[0] if industry in le_industry.classes_ else 0
    hq_enc = le_hq.transform([headquarters])[0] if headquarters in le_hq.classes_ else 0

    # Scale numerical features
    scaled_features = scaler.transform([[sales, assets, market_value]])[0]

    # Final input
    features = np.array([[scaled_features[0], scaled_features[1], scaled_features[2], industry_enc, hq_enc]])

    # Predict
    pred = model.predict(features)[0]
    return "High Profit" if pred == 1 else "Low Profit"


if __name__ == "__main__":
    # Example usage
    result = predict_profitability(
        sales=500, assets=1000, market_value=800, 
        industry="Banking", headquarters="United States"
    )
    print("Prediction:", result)
