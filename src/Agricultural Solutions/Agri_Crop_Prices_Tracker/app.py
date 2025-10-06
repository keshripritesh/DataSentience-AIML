from flask import Flask, render_template, request, jsonify, redirect
import requests
# New imports for Car Price Prediction
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle
import os
import datetime

app = Flask(__name__)


# Global API config
API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
API_PARAMS = {
    "api-key": "579b464db66ec23bdd000001c43ef34767ce496343897dfb1893102b",
    "format": "json",
    "limit": 1000
}


# Load data once (to avoid repeated API hits)
def load_data():
    response = requests.get(API_URL, params=API_PARAMS)
    return response.json().get("records", [])

DATA = load_data()


@app.route('/')
def home():
    return redirect('/crop_price_tracker')

@app.route('/crop_price_tracker', methods=['GET', 'POST'])
def crop_price_tracker():
    crops = sorted({record['commodity'] for record in DATA if record.get('commodity')})
    result = []
    error = None

    if request.method == 'POST':
        crop = request.form.get('crop')
        state = request.form.get('state')
        market = request.form.get('market')

        result = [
            r for r in DATA
            if r.get('commodity', '').lower() == crop.lower()
            and r.get('state', '').lower() == state.lower()
            and r.get('market', '').lower() == market.lower()
        ]

        if not result:
            error = "No data found for the given crop, state, and market."

    return render_template('crop_price_tracker.html', crops=crops, result=result, error=error)

@app.route('/get_states')
def get_states():
    crop = request.args.get('crop', '').lower()
    states = sorted({r['state'] for r in DATA if r.get('commodity', '').lower() == crop})
    return jsonify(states)

@app.route('/get_markets')
def get_markets():
    crop = request.args.get('crop', '').lower()
    state = request.args.get('state', '').lower()
    markets = sorted({
        r['market'] for r in DATA
        if r.get('commodity', '').lower() == crop and r.get('state', '').lower() == state
    })
    return jsonify(markets)


# --- CAR PRICE PREDICTOR ---
MODEL_FILE = 'car_price_model.pkl'
DATA_FILE = os.path.join('data', 'car_data.csv')

def train_and_save_model():
    """Trains the Ridge regression model and saves it to a file."""
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory. Please add 'car_data.csv' to it.")
        return None

    if not os.path.exists(DATA_FILE):
        print(f"Dataset not found at {DATA_FILE}. Please download it from Kaggle ('Car Dekho dataset') and place it there.")
        return None

    # Load and preprocess data
    df = pd.read_csv(DATA_FILE)
    df.drop(['Car_Name'], axis=1, inplace=True)

    # Feature Engineering: Car Age
    current_year = datetime.datetime.now().year
    df['no_year'] = current_year - df['Year']
    df.drop(['Year'], axis=1, inplace=True)

    # Define features and target
    X = df.drop(['Selling_Price'], axis=1)
    y = df['Selling_Price']

    # Create a column transformer for one-hot encoding categorical features
    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Fuel_Type', 'Seller_Type', 'Transmission']),
        remainder='passthrough'
    )

    # Create a Ridge regression pipeline for regularization
    ridge = Ridge(alpha=1.0, random_state=42)
    pipe = make_pipeline(column_trans, ridge)

    # Train the model on the entire dataset
    pipe.fit(X, y)

    # Save the model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(pipe, f)

    print("Model trained and saved successfully.")
    return pipe

# Try to load the model, or train it if it doesn't exist
try:
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
    else:
        model = train_and_save_model()
except (FileNotFoundError, EOFError, pickle.UnpicklingError):
    model = train_and_save_model()

@app.route('/car_price_predictor', methods=['GET', 'POST'])
def car_price_predictor():
    prediction = None
    error = None
    # These values should match the ones in the dataset
    fuel_types = ['Petrol', 'Diesel', 'CNG']
    seller_types = ['Dealer', 'Individual']
    transmission_types = ['Manual', 'Automatic']

    if not model:
        error = "Model is not trained. Please add 'car_data.csv' to the 'data' directory and restart the server."

    if request.method == 'POST' and model:
        try:
            # Create a DataFrame from the form input
            input_data = pd.DataFrame({
                'Present_Price': [float(request.form['present_price'])],
                'Kms_Driven': [int(request.form['kms_driven'])],
                'Fuel_Type': [request.form['fuel_type']],
                'Seller_Type': [request.form['seller_type']],
                'Transmission': [request.form['transmission']],
                'Owner': [int(request.form['owner'])],
                'no_year': [datetime.datetime.now().year - int(request.form['year'])]
            })

            predicted_price = model.predict(input_data)[0]
            prediction = f"Predicted Selling Price: ₹{predicted_price:.2f} Lakhs"

        except Exception as e:
            error = f"An error occurred during prediction: {e}"

    return render_template('car_price_predictor.html',
                           prediction=prediction, error=error, fuel_types=fuel_types,
                           seller_types=seller_types, transmission_types=transmission_types)

if __name__ == '__main__':
    app.run(debug=True, port=5001)