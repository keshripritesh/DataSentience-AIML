import os
import pandas as pd
from typing import Tuple

DATA_PATH = "data/stock_ai_generated.csv"

NUMERIC_COLS = ["R&D_Spending_USD_Mn", "AI_Revenue_USD_Mn", "AI_Revenue_Growth_%"]
CAT_COLS = ["Company"]
TEXT_COL = "Event"
DATE_COL = "Date"
TARGET_COL = "Stock_Impact_%"

def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleanup
    df = df.copy()
    # Datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    # Numeric coercion
    for c in NUMERIC_COLS + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Event text
    if TEXT_COL in df.columns:
        df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
    else:
        df[TEXT_COL] = ""
    # Drop rows missing critical fields
    df = df.dropna(subset=[DATE_COL] + NUMERIC_COLS + [TARGET_COL])
    # Date-derived features
    df["Month"] = df[DATE_COL].dt.month
    df["DayOfWeek"] = df[DATE_COL].dt.dayofweek  # 0=Mon
    # Final ordering (keep what train.py expects)
    return df

def preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = _prepare_frame(df)

    # Chronological split: last 20% dates → test
    df = df.sort_values(DATE_COL)
    cutoff_idx = int(len(df) * 0.8)
    train_df = df.iloc[:cutoff_idx].copy()
    test_df  = df.iloc[cutoff_idx:].copy()

    feature_cols = NUMERIC_COLS + CAT_COLS + ["Month", "DayOfWeek", TEXT_COL]

    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df[TARGET_COL].reset_index(drop=True)
    X_test  = test_df[feature_cols].reset_index(drop=True)
    y_test  = test_df[TARGET_COL].reset_index(drop=True)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    print("✅ Preprocessing complete")
    print("Train size:", X_train.shape, " Test size:", X_test.shape)
