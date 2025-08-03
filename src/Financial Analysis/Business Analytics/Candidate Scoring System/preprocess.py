# preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def map_text_to_score(text):
    if pd.isnull(text): return 0
    text = text.lower()
    if 'good' in text or 'impactful' in text:
        return 3
    elif 'guarded' in text:
        return 2
    elif 'struggled' in text or 'not able' in text or 'unable' in text:
        return 1
    return 0

def preprocess_data(df, is_train=True):
    df = df.copy()

    # ✅ Clean column names
    df.columns = df.columns.str.strip()

    # Drop rows without Total Score if training
    if is_train:
        df = df[df["Total Score"].notna()]

    # Clean and convert experience
    df["Experienced candidate - (Experience in months)"] = pd.to_numeric(
        df["Experienced candidate - (Experience in months)"], errors="coerce"
    ).fillna(0)

    # Map text-based ordinal features
    text_features = [
        "Confidence based on Introduction (English)",
        "Confidence based on the topic given",
        "Structured Thinking (In regional only)",
        "Regional fluency based on the topic given"
    ]

    for col in text_features:
        df[col] = df[col].apply(map_text_to_score)

    # Encode mode of interview
    df["Mode of interview given by candidate?"] = LabelEncoder().fit_transform(
        df["Mode of interview given by candidate?"].astype(str)
    )

    features = text_features + [
        "Mode of interview given by candidate?",
        "Experienced candidate - (Experience in months)"
    ]

    X = df[features]
    y = df["Total Score"] if is_train else None

    return X, y
