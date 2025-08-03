import pandas as pd

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=['skills', 'email'], inplace=True)
    df['skills'] = df['skills'].str.lower().str.replace(',', ' ')
    return df
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_and_encode(df):
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate columns

    drop_cols = ['Comments', 'RedFlags Comments in Interview']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Extract target column if present (preserve it from encoding)
    target_col = 'Whether joined the company or not'
    target = df[target_col] if target_col in df.columns else None

    # Drop target temporarily from features
    if target is not None:
        df = df.drop(columns=[target_col])

    # Encode all categorical features
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Add back the target column
    if target is not None:
        df[target_col] = target

    return df
