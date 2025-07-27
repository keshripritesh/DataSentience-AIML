import pickle
from preprocess import clean_text

# Load model
with open("models/spam_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

def predict_spam(comment):
    cleaned = clean_text(comment)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    return "Spam" if pred == 1 else "Not Spam"

if __name__ == "__main__":
    while True:
        text = input("\nEnter a YouTube comment (or type 'exit'): ")
        if text.lower() == 'exit':
            break
        print("Prediction:", predict_spam(text))
