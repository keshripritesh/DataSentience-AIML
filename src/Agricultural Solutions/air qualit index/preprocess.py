import pandas as pd

FEATURES = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

def preprocess_data(df, is_train=True):
    df = df.copy()

    # Drop non-feature columns
    if "Date" in df.columns: df.drop(columns=["Date"], inplace=True)
    if "City" in df.columns: df.drop(columns=["City"], inplace=True)
    if "AQI_Bucket" in df.columns: df.drop(columns=["AQI_Bucket"], inplace=True)

    # Handle missing pollutants
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # If training: remove rows with missing AQI
    if is_train:
        df = df.dropna(subset=["AQI"])

    X = df[FEATURES]
    y = df["AQI"] if "AQI" in df.columns else None
    return X, y
