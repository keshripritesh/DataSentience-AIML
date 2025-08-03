# 🏨 Airbnb Booking Demand Classifier

Predict whether an Airbnb listing in New York City is likely to be frequently booked based on features like location, pricing, room type, and review activity.

[!ui screenshot](assets/image.png)
---

## 📌 Project Overview

This project is a **binary classification task** that uses machine learning to determine if an Airbnb listing is **"frequently booked"**. A listing is labeled as frequently booked if it receives **more than 1 review per month** on average.

---

## ✅ Problem Statement

Airbnb hosts and the platform can benefit from understanding what makes a listing successful. This model helps classify listings into:
- **Frequently Booked** (`reviews_per_month > 1`)
- **Infrequently Booked** (`reviews_per_month <= 1`)

---

## 🧠 ML Approach

- **Model Type:** Binary Classification  
- **Target Variable:** `is_frequently_booked` (1 = yes, 0 = no)  
- **Algorithms Used:** Random Forest Classifier (default), but modular for others (e.g., Logistic Regression, SVM)

---

## 🗂️ Project Structure

airbnb-booking-classifier/
│
├── data/
│ └── airbnb_nyc.csv # Raw dataset
│
├── model/
│ ├── booking_model.pkl # Trained classifier
│ └── preprocessor.pkl # Preprocessing pipeline (optional)
│
├── preprocess.py # Data cleaning and preprocessing logic
├── train.py # Trains model and saves it
├── predict.py # Loads model and predicts on new data
├── README.md # Project overview and usage

markdown
Copy
Edit

---

## 🧼 Preprocessing

Handled in `preprocess.py`, which:
- Drops entries with missing `reviews_per_month`
- Creates `is_frequently_booked` binary target
- One-hot encodes categorical features
- Fills missing numerical values with median

Used features:
- `neighbourhood_group`
- `room_type`
- `price`
- `minimum_nights`
- `number_of_reviews`
- `availability_365`
- `calculated_host_listings_count`

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/airbnb-booking-classifier.git
cd airbnb-booking-classifier
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is missing, install key packages:

bash
Copy
Edit
pip install pandas scikit-learn joblib
3. Add your data
Place the airbnb_nyc.csv dataset into the data/ folder.

4. Train the model
bash
Copy
Edit
python train.py
This will output evaluation metrics and save the model to model/booking_model.pkl.

5. Run prediction
Update the sample_input in predict.py and run:

bash
Copy
Edit
python predict.py
Example output:

json
Copy
Edit
{
  "prediction": "Frequently Booked",
  "confidence": 0.88
}
📊 Example: Sample Input for Prediction
python
Copy
Edit
sample_input = {
    "neighbourhood_group": "Manhattan",
    "room_type": "Entire home/apt",
    "price": 150,
    "minimum_nights": 3,
    "number_of_reviews": 50,
    "availability_365": 200,
    "calculated_host_listings_count": 2