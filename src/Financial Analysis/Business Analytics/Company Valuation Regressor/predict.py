import joblib
import numpy as np

MODEL_PATH = "model/company_valuation_model.pkl"

def predict_company_valuation(sales, profit, assets, industry, hq):
    # Load saved objects
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    scaler = saved["scaler"]
    le_industry = saved["le_industry"]
    le_hq = saved["le_hq"]

    # Encode categorical
    industry_encoded = le_industry.transform([industry])[0]
    hq_encoded = le_hq.transform([hq])[0]

    # Prepare features
    features = np.array([[sales, profit, assets, industry_encoded, hq_encoded]])
    features_scaled = scaler.transform(features)

    # Prediction
    pred = model.predict(features_scaled)[0]
    return "High Valuation" if pred == 1 else "Low Valuation"


if __name__ == "__main__":
    # Example prediction
    result = predict_company_valuation(
        sales=500, profit=80, assets=700, industry="Banking", hq="United States"
    )
    print("Predicted Valuation Category:", result)
