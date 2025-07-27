# preprocess.py

import pandas as pd

def preprocess_data(input_path='data/Rainfall.csv', output_path='cleaned_data.csv'):
    df = pd.read_csv(input_path)

    # Strip and lowercase column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Encode 'rainfall' (yes → 1, no → 0)
    df['rainfall'] = df['rainfall'].map({'yes': 1, 'no': 0})

    # Drop missing rows (e.g., winddirection, windspeed)
    df = df.dropna()

    df.to_csv(output_path, index=False)
    print(f"[✅] Cleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    preprocess_data()
