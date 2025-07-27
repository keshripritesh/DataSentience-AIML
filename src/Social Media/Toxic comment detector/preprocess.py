import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(subset=["comment_text"], inplace=True)
    X = df["comment_text"].values
    y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
    return X, y

def preprocess(X, vectorizer=None, fit=False):
    if fit:
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        X = vectorizer.fit_transform(X)
    else:
        X = vectorizer.transform(X)
    return X, vectorizer
