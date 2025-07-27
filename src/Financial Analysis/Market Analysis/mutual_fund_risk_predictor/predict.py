# predict.py

import pandas as pd
import joblib

def load_model(model_path='models/rf_model.pkl'):
    model, label_encoder = joblib.load(model_path)
    return model, label_encoder

def preprocess_input(input_df):
    # Clean numeric fields
    input_df['1 month return'] = input_df['1 month return'].str.replace('%', '', regex=False).astype(float)
    input_df['1 Year return'] = input_df['1 Year return'].str.replace('%', '', regex=False).astype(float)
    input_df['3 Year Return'] = pd.to_numeric(input_df['3 Year Return'].str.replace('%', ''), errors='coerce')
    input_df['Minimum investment'] = input_df['Minimum investment'].str.replace('Rs.', '', regex=False).str.replace(',', '', regex=False).astype(float)
    input_df['AUM'] = input_df['AUM'].str.replace(' cr', '', regex=False).str.replace(',', '', regex=False)
    input_df['AUM'] = pd.to_numeric(input_df['AUM'], errors='coerce')

    # Drop unnecessary columns if present
    input_df = input_df.drop(['Fund Name', 'Fund Manager'], axis=1, errors='ignore')

    # One-hot encode
    input_df = pd.get_dummies(input_df, columns=['AMC', 'Category'], drop_first=True)

    return input_df

def align_columns(input_df, reference_columns):
    for col in reference_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # add missing cols with zero
    input_df = input_df[reference_columns]  # align column order
    return input_df

def predict_risk(input_data, model_path='models/rf_model.pkl', reference_csv='cleaned_data.csv'):
    # Load model and encoder
    model, label_encoder = load_model(model_path)

    # Load reference structure
    ref_df = pd.read_csv(reference_csv)
    reference_columns = ref_df.drop(['Risk'], axis=1).columns.tolist()

    # Prepare input
    input_df = pd.DataFrame([input_data])
    input_df = preprocess_input(input_df)
    input_df = align_columns(input_df, reference_columns)

    # Predict
    prediction = model.predict(input_df)
    risk_label = label_encoder.inverse_transform(prediction)[0]
    return risk_label

if __name__ == "__main__":
    # Sample mutual fund input
    sample_input = {
        "AMC": "mahindra manulife mutual fund",
        "Fund Name": "Demo Fund",
        "Morning star rating": 3,
        "Value Research rating": 4,
        "1 month return": "5.50%",
        "NAV": 25.6,
        "1 Year return": "42.50%",
        "3 Year Return": "19.20%",
        "Minimum investment": "Rs.500.0",
        "Fund Manager": "Test Manager",
        "AUM": "123.45 cr",
        "Category": "Equity"
    }

    risk = predict_risk(sample_input)
    print(f"[🔮] Predicted Risk Category: {risk}")
