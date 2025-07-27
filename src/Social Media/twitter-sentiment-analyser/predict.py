import pickle
from preprocess import clean_tweet

with open("models/sentiment_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

LABELS = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def predict_sentiment(text):
    cleaned = clean_tweet(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return LABELS[pred]

if __name__ == "__main__":
    tweet = input("Enter tweet: ")
    sentiment = predict_sentiment(tweet)
    print(f"Predicted Sentiment: {sentiment}")
