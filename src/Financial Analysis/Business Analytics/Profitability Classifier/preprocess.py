import pandas as pd
import csv
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = "data/Forbes_2000_Companies_2025.csv"

def clean_numeric(val):
    """Clean messy numeric strings like '0.513.7' -> '513.7'."""
    if pd.isna(val):
        return None
    val = str(val).replace(",", "").strip()
    # Keep only first decimal point
    if val.count(".") > 1:
        # Remove all dots except the last one before decimals
        parts = re.split(r"\.", val)
        val = parts[0] + "." + "".join(parts[1:])
    try:
        return float(val)
    except:
        return None

def preprocess_data():
    # Detect delimiter
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        dialect = csv.Sniffer().sniff(sample)

    # Load dataset
    df = pd.read_csv(DATA_PATH, delimiter=dialect.delimiter, engine="python")

    # Clean numeric columns
    for col in ["Sales ($B)", "Profit ($B)", "Assets ($B)", "Market Value ($B)"]:
        df[col] = df[col].apply(clean_numeric)

    # Drop rows with missing/invalid numeric values
    df.dropna(subset=["Sales ($B)", "Profit ($B)", "Assets ($B)", "Market Value ($B)"], inplace=True)

    # Target variable: classify as High Profit (>= median) or Low Profit
    median_profit = df["Profit ($B)"].median()
    df["Profit_Class"] = df["Profit ($B)"].apply(lambda x: 1 if x >= median_profit else 0)

    # Features
    X = df[["Sales ($B)", "Assets ($B)", "Market Value ($B)", "Industry", "Headquarters"]]
    y = df["Profit_Class"]

    # Encode categorical features
    le_industry = LabelEncoder()
    le_hq = LabelEncoder()
    X["Industry"] = le_industry.fit_transform(X["Industry"])
    X["Headquarters"] = le_hq.fit_transform(X["Headquarters"])

    # Scale numerical features
    scaler = StandardScaler()
    X[["Sales ($B)", "Assets ($B)", "Market Value ($B)"]] = scaler.fit_transform(
        X[["Sales ($B)", "Assets ($B)", "Market Value ($B)"]]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, le_industry, le_hq, scaler
