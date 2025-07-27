# preprocess.py

import pandas as pd

def preprocess_data(input_path='data/survey lung cancer.csv', output_path='cleaned_data.csv'):
    df = pd.read_csv(input_path)

    # Strip column names and normalize
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]

    # Encode 'GENDER' and 'LUNG_CANCER'
    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

    # Ensure all columns are numeric
    df = df.dropna()
    df = df.astype(int)

    df.to_csv(output_path, index=False)
    print(f"[✅] Cleaned data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data()
