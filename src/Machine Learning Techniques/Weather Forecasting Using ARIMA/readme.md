## 🔍 Project Overview
This project focuses on time-series forecasting of weather data (specifically temperature) using the ARIMA (AutoRegressive Integrated Moving Average) model. With a strong emphasis on backend modeling and frontend usability, this app leverages the Streamlit framework to offer a sleek, interactive interface for real-time weather forecasting.

## 📈 Objective
To forecast future temperature values using historical weather data with an ARIMA model, and provide an easy-to-use web interface for visualization and interaction.

---

## 📊 Dataset Used
We use a publicly available [weather dataset on Kaggle by sumanthvrao](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data). The dataset typically contains:
- Date/Time
- Temperature (Celsius/Fahrenheit)
- Humidity
- Pressure

For this version, we focus on **daily average temperature**.

### Sample Columns:
| Date       | Temperature (°C) |
|------------|------------------|
| 2020-01-01 | 23.5             |
| 2020-01-02 | 24.1             |

---

## 🔧 Tech Stack / Libraries
- **Python 3.8+**
- **Pandas**: Data loading & manipulation
- **Numpy**: Numerical operations
- **Matplotlib & Seaborn**: Visualizations
- **Scikit-learn**: Evaluation metrics (RMSE)
- **Statsmodels**: SARIMAX implementation, ACF and PACF plotter, and Augmented Dickey Fuller Test(to check stationarity).
- **Streamlit**: Interactive web UI

---

## 🔁 Workflow

### 1. Data Collection & Cleaning
- Load weather data from CSV
- Convert date to datetime format
- Handle missing values (e.g., imputation or interpolation)
- Resample to daily average temperature if needed
- Since the Data was daily, SARIMAX took a long time to fit it
  So instead, we averaged the temperature over a week to shorten the dataset
  and also to reduce the fitting time for SARIMAX.

### 2. Exploratory Data Analysis (EDA)
- Line plot of temperature over time
- Decomposition of time series (trend, seasonality)
- ACF and PACF plots to analyze lags to make the Time Series stationary.

### 3. Model Building
- Use `statsmodels.tsa.statespace.sarimax.SARIMAX`
- Use PACF, ACF Plots to decide values for p,d,q, and P, D, Q(Seasonal)
- Train-test split (e.g., last 30 days as test set)

### 4. Forecasting
- Predict future values (n-steps ahead)
- Plot observed vs predicted values
- Evaluate using **Root Mean Square Error (RMSE)**
- The temperature for the past weeks over Delhi was extracted from [The Free Weather API on WeatherAPI.com](https://www.weatherapi.com/)
- If you want to run the predictions for the past weeks, make sure to create an account at the Waether API Website
  and insert your API Key in the placeholder `API_KEY`

### 5. Streamlit Interface
- Sidebar for parameter input
- Buttons to run forecast
- Plotly charts for interactive graphs

### 6. Documentation & Packaging
- Provide Jupyter Notebook for exploration
- Python script for deployment
- Requirements.txt for dependencies
- README file (this)

---

## 📉 Model Evaluation

We use **Root Mean Square Error (RMSE)** as our primary metric:

