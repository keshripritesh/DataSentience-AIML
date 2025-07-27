import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.preprocessing import load_and_preprocess_data
from models.arima_model import train_arima_model, forecast_arima
from models.lstm_model import prepare_lstm_data, train_lstm

# App config
st.set_page_config(page_title="💱 Currency Forecasting (ARIMA & LSTM)", layout="wide")

# App Header
st.title("💱 Currency Exchange Rate Forecasting")
st.markdown("""
This interactive dashboard forecasts currency exchange rates using:
- **ARIMA** for traditional time series prediction
- **LSTM** for deep learning-based forecasting
""")

st.markdown("---")

# Sidebar Filters
st.sidebar.header("🔧 Configuration")

currency = st.sidebar.selectbox("Choose Currency", ["Indian Rupee", "US Dollar", "Euro", "British Pound"])
model_option = st.sidebar.radio("Forecasting Model", ["ARIMA", "LSTM"])
show_forecast = st.sidebar.button("📈 Run Forecast")

# Load Data
df = load_and_preprocess_data("data/cleaned_exchange_rates.csv", currency=currency)

# Show data preview
with st.expander("🔍 View Raw Exchange Rate Data"):
    st.dataframe(df.tail(10), use_container_width=True)

# Forecasting
if show_forecast:
    st.markdown("---")

    if model_option == "ARIMA":
        st.subheader("📊 ARIMA Forecast")
        model = train_arima_model(df['Exchange_Rate'])
        prediction = forecast_arima(model, steps=10)

        st.write("📆 Forecast for Next 10 Days")
        st.dataframe(prediction.rename("Forecast").reset_index(), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        df['Exchange_Rate'].plot(ax=ax, label="Historical", color="blue")
        prediction.plot(ax=ax, label="Forecast", color="red")
        ax.set_title(f"{currency} Forecast using ARIMA")
        ax.set_xlabel("Date")
        ax.set_ylabel("Exchange Rate")
        ax.legend()
        st.pyplot(fig)

    elif model_option == "LSTM":
        st.subheader("🧠 LSTM Forecast")

        with st.spinner("Training LSTM model..."):
            X, y, scaler = prepare_lstm_data(df['Exchange_Rate'])
            model = train_lstm(X, y)

        y_pred_scaled = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_true = scaler.inverse_transform(y.reshape(-1, 1))

        st.write("🆚 Last 10 Predictions vs Actual")
        comparison = pd.DataFrame({
            "Actual": [round(float(a[0]), 4) for a in y_true[-10:]],
            "Predicted": [round(float(p[0]), 4) for p in y_pred[-10:]]
        })
        st.table(comparison)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_true, label="Actual", color="green")
        ax.plot(y_pred, label="Predicted", color="orange")
        ax.set_title(f"{currency} Forecast using LSTM")
        ax.set_xlabel("Time")
        ax.set_ylabel("Exchange Rate")
        ax.legend()
        st.pyplot(fig)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Models: ARIMA & LSTM")
