🏠 Optimal Pricing Recommendation System

A machine learning model that recommends optimal listing prices for Airbnb NYC rentals based on features like location, room type, availability, and reviews.
[!ui screenshot](assets/image.png)
---

## 📌 Objective

The goal of this project is to **predict the ideal nightly price** for an Airbnb listing using historical data. By understanding what features influence pricing, hosts can make data-driven decisions to improve revenue and competitiveness.

---

## 📂 Project Structure

optimal-pricing-recommender/
│
├── data/
│ └── airbnb_nyc.csv # Raw dataset
│
├── model/
│ └── price_model.pkl # Saved trained model
│
├── preprocess.py # Data cleaning and transformation
├── train.py # Model training using Random Forest
├── predict.py # Inference script
└── README.md # Project documentation

markdown
Copy
Edit

---

## 🧠 Model Overview

- **Model Type:** Regression
- **Algorithm:** Random Forest Regressor
- **Target Variable:** `price` (log-transformed)
- **Features Used:**
  - `neighbourhood_group`
  - `room_type`
  - `latitude`, `longitude`
  - `minimum_nights`
  - `availability_365`
  - `number_of_reviews`
  - `reviews_per_month`
  - `calculated_host_listings_count`

---

## 🛠️ How It Works

1. **Preprocessing**
   - Filters out price outliers
   - Fills missing values
   - One-hot encodes categorical features
   - Applies log transform on `price` to reduce skew

2. **Training**
   - Random Forest with 200 trees and max depth of 15
   - Model evaluated using RMSE and R² score

3. **Prediction**
   - Accepts a new listing's features
   - Returns an optimal price recommendation in USD

---

## 🚀 Usage

### 📌 Step 1: Install Requirements

```bash
pip install pandas scikit-learn joblib numpy
📌 Step 2: Train the Model
bash
Copy
Edit
python train.py
📌 Step 3: Predict Price
Edit predict.py or use from a script:

python
Copy
Edit
from predict import predict_price

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
📊 Example Output
yaml
Copy
Edit
RMSE: $45.23
R² Score: 0.6125
Recommended Price: $88.41