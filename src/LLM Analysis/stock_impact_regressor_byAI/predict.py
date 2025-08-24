import argparse
import joblib
import pandas as pd
from datetime import datetime

MODEL_PATH = "model/stock_impact_regressor.pkl"

def predict(company, rd, revenue, growth, event):
    # Load model
    model = joblib.load(MODEL_PATH)

    # Construct input DataFrame with SAME column names as training
    input_data = pd.DataFrame([{
        "Company": company,
        "R&D_Spending_USD_Mn": rd,
        "AI_Revenue_USD_Mn": revenue,
        "AI_Revenue_Growth_%": growth,
        "Event": event,
        "Date": datetime.now().strftime("%Y-%m-%d")   # dummy or current date
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict stock impact of AI events")
    parser.add_argument("--company", type=str, required=True, help="Company name")
    parser.add_argument("--rd", type=float, required=True, help="R&D Spending (USD Mn)")
    parser.add_argument("--revenue", type=float, required=True, help="AI Revenue (USD Mn)")
    parser.add_argument("--growth", type=float, required=True, help="AI Revenue Growth %")
    parser.add_argument("--event", type=str, required=True, help="Event description")

    args = parser.parse_args()

    impact = predict(args.company, args.rd, args.revenue, args.growth, args.event)
    print(f"Predicted Stock Impact: {impact:.2f}")
