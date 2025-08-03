# preprocess.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np

def preprocess_data(df, fit_vectorizer=True):
    # Drop unnecessary columns
    df = df.drop(columns=["Company Name"])

    # Separate sentiment for embedding
    sentiments = df["Employee Sentiment"]
    other_features = df.drop(columns=["GenAI Tool", "Employee Sentiment"])
    
    # One-hot encode categorical features
    # One-hot encode categorical features
    categorical_cols = ["Industry", "Country"]
    encoder = OneHotEncoder(sparse_output=False,    handle_unknown="ignore")
    encoded_cat = encoder.fit_transform(other_features[categorical_cols])


    # Normalize numerical features
    numerical_cols = [col for col in other_features.columns if col not in categorical_cols]
    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(other_features[numerical_cols])
    
    # Sentiment embeddings via TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    sentiment_embed = vectorizer.fit_transform(sentiments).toarray()

    # Combine features
    X = np.concatenate([encoded_cat, scaled_num, sentiment_embed], axis=1)
    y = df["GenAI Tool"]

    # Save vectorizers
    if fit_vectorizer:
        joblib.dump(encoder, "model/encoder.pkl")
        joblib.dump(scaler, "model/scaler.pkl")
        joblib.dump(vectorizer, "model/vectorizer.pkl")

    return X, y

def preprocess_input(sample_df):
    encoder = joblib.load("model/encoder.pkl")
    scaler = joblib.load("model/scaler.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")

    sample_sentiment = sample_df["Employee Sentiment"]
    sample_features = sample_df.drop(columns=["Employee Sentiment"])

    encoded_cat = encoder.transform(sample_features[["Industry", "Country"]])
    scaled_num = scaler.transform(sample_features.drop(columns=["Industry", "Country"]))
    sentiment_embed = vectorizer.transform(sample_sentiment).toarray()

    return np.concatenate([encoded_cat, scaled_num, sentiment_embed], axis=1)
