# preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, is_train=True):
    df = df.copy()

    # Drop rows with missing target during training
    if is_train:
        df = df[df["Interview Verdict"].notna()]

    # Select useful columns
    cols = [
        "Type of Graduation/Post Graduation",
        "Mode of interview given by candidate?",
        "Gender",
        "Experienced candidate - (Experience in months)",
        "Confidence Score",
        "Structured Thinking Score",
        "Regional Fluency Score",
        "Total Score"
    ]
    
    if is_train:
        cols.append("Interview Verdict")

    df = df[cols]

    # Fill missing experience with 0
    df["Experienced candidate - (Experience in months)"] = pd.to_numeric(
        df["Experienced candidate - (Experience in months)"], errors="coerce"
    ).fillna(0)

    # Encode categorical features
    cat_cols = [
        "Type of Graduation/Post Graduation",
        "Mode of interview given by candidate?",
        "Gender"
    ]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Encode target if training
    target_encoder = None
    if is_train:
        target_encoder = LabelEncoder()
        df["Interview Verdict"] = target_encoder.fit_transform(df["Interview Verdict"])
    
    X = df.drop("Interview Verdict", axis=1) if is_train else df
    y = df["Interview Verdict"] if is_train else None

    return X, y, encoders, target_encoder
