
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def analyze_sentiment(texts, tokenizer, model):
    labels = ['negative', 'neutral', 'positive']
    results = {l: 0 for l in labels}

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = labels[torch.argmax(scores)]
        results[pred] += 1

    total = sum(results.values())
    return {k: round((v / total) * 100, 2) for k, v in results.items()}
