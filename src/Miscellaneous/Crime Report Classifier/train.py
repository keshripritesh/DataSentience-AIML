
# train.py
"""
Train a text classifier to predict whether an article title is a crime report.
Saves a scikit-learn Pipeline (TfidfVectorizer + LogisticRegression) as a single artifact.
Usage:
    python train.py --csv /path/to/CrimeVsNoCrimeArticles.csv --model /path/to/model.joblib
"""

import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from preprocess import load_data, get_X_y, TEXT_COL, LABEL_COL

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None))
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/CrimeVsNoCrimeArticles.csv", help="Path to CrimeVsNoCrimeArticles.csv (default: data/CrimeVsNoCrimeArticles.csv)")
    parser.add_argument("--model", type=str, default="model/crime_model.joblib", help="Path to save trained model joblib (default: model/crime_model.joblib)")
    args = parser.parse_args()

    df = load_data(args.csv, text_col=TEXT_COL, label_col=LABEL_COL)
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(pipe, args.model)
    print(f"\nModel saved to: {args.model}")

if __name__ == "__main__":
    main()
