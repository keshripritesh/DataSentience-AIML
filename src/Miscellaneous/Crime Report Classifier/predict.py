
# predict.py
"""
Load the trained model and run predictions on input text.
Usage:
    # Single text
    python predict.py --model /path/to/model.joblib --text "robbery reported near downtown mall"
    # From a file (one title per line)
    python predict.py --model /path/to/model.joblib --file /path/to/titles.txt
"""
###python predict.py --text "Police investigate burglary in city center"
###python predict.py --text "New park opens in residential neighborhood" use statemens like this to test it

import argparse
import sys
import joblib
from typing import List
from preprocess import clean_text

def load_model(path: str):
    return joblib.load(path)

def predict_texts(model, texts: List[str]) -> List[int]:
    cleaned = [clean_text(t) for t in texts]
    preds = model.predict(cleaned)
    return preds.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model/crime_model.joblib", help="Path to trained joblib model (default: model/crime_model.joblib)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="A single text/title to classify")
    group.add_argument("--file", type=str, help="Path to a text file with one title per line")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.text:
        texts = [args.text]
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    preds = predict_texts(model, texts)
    for t, p in zip(texts, preds):
        label = "CRIME" if p == 1 else "NO_CRIME"
        print(f"{label}\t{t}")

if __name__ == "__main__":
    main()
