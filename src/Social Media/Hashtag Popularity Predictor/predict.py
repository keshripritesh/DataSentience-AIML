import pickle
import pandas as pd
from preprocess import clean_hashtag

# Load preprocessor and model
with open("models/popularity_model.pkl", "rb") as f:
    preprocessor, model = pickle.load(f)

def predict_popularity(hashtag, mentions, sentiment_score):
    cleaned = clean_hashtag(hashtag)
    
    # Wrap input into a DataFrame with correct column names
    data = pd.DataFrame([{
        'hashtag': cleaned,
        'mentions': mentions,
        'sentiment_score': sentiment_score
    }])
    
    X = preprocessor.transform(data)
    prediction = model.predict(X)[0]
    return "Popular ✅" if prediction == 1 else "Not Popular ❌"

if __name__ == "__main__":
    tag = input("Enter a hashtag: ")
    mentions = int(input("Mention count: "))
    sentiment = float(input("Sentiment score (e.g., 0.65): "))
    
    result = predict_popularity(tag, mentions, sentiment)
    print("📊 Prediction:", result)
