# Agri-AI Solutions

A **Flask-based web application** that provides various AI-powered solutions.

---

## 🚀 Features Implemented

### 1. Crop Price Tracker
- Fetches live crop price data from the [Government of India Open Data API](https://data.gov.in).
- Search crop prices by **Crop Name**, **State**, and **Market**.
- Dynamic **State** and **Market** dropdowns for easy filtering.

### 2. Car Price Predictor
- Predicts the selling price of a car using a machine learning model.
- Uses a **Ridge Regression** model for improved accuracy and to prevent overfitting.
- Features used for prediction: `Year of Manufacture`, `Present Price`, `Kms Driven`, `Fuel Type`, `Seller Type`, `Transmission`, and `Number of Owners`.

---

## 🛠 Tech Stack
- **Python** (Flask)
- **Pandas** for data manipulation
- **Scikit-learn** for machine learning
- **HTML/CSS** (Jinja2 templates)
- **JavaScript** (AJAX for dynamic dropdowns)
- **Government Open Data API**

---

## 📦 Setup and Installation

1. **Clone the repository**
   
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Car Price Dataset**
   - Create a `data` directory in the root of the project.
   - Download the dataset from Kaggle: Car Dekho Used Car Data.
   - Place the `car data.csv` file inside the `data` directory. The final path should be `data/car_data.csv`.

5. **Run the application**
   ```bash
   python app.py
   ```
   The application will be available at `http://127.0.0.1:5001`. The first time you run it, it will train and save the car price prediction model (`car_price_model.pkl`).

---

## 🌐 Endpoints
- **Crop Price Tracker**: `http://127.0.0.1:5001/crop_price_tracker`
- **Car Price Predictor**: `http://127.0.0.1:5001/car_price_predictor`
