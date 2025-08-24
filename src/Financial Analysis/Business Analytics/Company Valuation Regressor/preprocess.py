import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = "data/companies.csv"

def preprocess_data():
    # Load dataset (semicolon separated)
    df = pd.read_csv(DATA_PATH, sep=";")

    # Strip extra spaces from column names
    df.columns = df.columns.str.strip()

    # Convert numeric columns (remove commas + spaces before casting)
    numeric_cols = ["Sales ($B)", "Profit ($B)", "Assets ($B)", "Market Value ($B)"]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)  # remove commas
            .str.replace(" ", "", regex=False)  # remove spaces
        )
        # Force errors='coerce' so bad values turn into NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN after cleaning
    df = df.dropna(subset=numeric_cols)

    # Create target variable: High vs Low Market Value
    median_value = df["Market Value ($B)"].median()
    df["ValuationCategory"] = df["Market Value ($B)"].apply(
        lambda x: 1 if x >= median_value else 0
    )

    # Encode categorical features
    le_industry = LabelEncoder()
    le_hq = LabelEncoder()

    df["Industry"] = le_industry.fit_transform(df["Industry"])
    df["Headquarters"] = le_hq.fit_transform(df["Headquarters"])

    # Features and target
    X = df[["Sales ($B)", "Profit ($B)", "Assets ($B)", "Industry", "Headquarters"]]
    y = df["ValuationCategory"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, le_industry, le_hq
