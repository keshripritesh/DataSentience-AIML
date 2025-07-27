# preprocess.py

import pandas as pd
import os

def clean_percentage(col):
    return col.str.replace('%', '', regex=False).astype(float)

def clean_investment(col):
    return col.str.replace('Rs.', '', regex=False).str.replace(',', '', regex=False).astype(float)

def clean_aum(col):
    col = col.str.replace(' cr', '', regex=False).str.replace(',', '', regex=False)
    return pd.to_numeric(col, errors='coerce')  # convert invalid strings to NaN

def preprocess_data(input_path, output_path='cleaned_data.csv'):
    print("[🔍] Reading dataset...")
    df = pd.read_csv(input_path)

    # Drop irrelevant or high-cardinality columns
    df = df.drop(['Fund Name', 'Fund Manager'], axis=1, errors='ignore')

    print("[🧹] Cleaning percentage columns...")
    df['1 month return'] = clean_percentage(df['1 month return'])
    df['1 Year return'] = clean_percentage(df['1 Year return'])
    df['3 Year Return'] = pd.to_numeric(df['3 Year Return'].str.replace('%', ''), errors='coerce')

    print("[💰] Cleaning investment and AUM fields...")
    df['Minimum investment'] = clean_investment(df['Minimum investment'])
    df['AUM'] = clean_aum(df['AUM'])

    print("[🔠] Encoding categorical columns...")
    df = pd.get_dummies(df, columns=['AMC', 'Category'], drop_first=True)

    print("[🧯] Dropping rows with missing target or corrupt numeric values...")
    df = df.dropna(subset=['Risk'])  # Drop if target is missing
    df = df.dropna()  # Drop any rows with NaNs due to conversion issues

    print(f"[💾] Saving cleaned data to {output_path}")
    df.to_csv(output_path, index=False)
    print("[✅] Preprocessing complete!")

if __name__ == "__main__":
    # Adjust the path if running from a different directory
    preprocess_data('data/Mutual_fund Data.csv')
