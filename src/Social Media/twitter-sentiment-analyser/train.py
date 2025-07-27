import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from preprocess import clean_tweet

# Load dataset
df = pd.read_csv("data/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'tweet']

# Preprocess
df['cleaned'] = df['tweet'].apply(clean_tweet)
df = df[df['cleaned'].str.strip().astype(bool)]  # remove empty tweets

# Map target (0 = negative, 2 = neutral, 4 = positive)
df['label'] = df['target'].map({0: 0, 2: 1, 4: 2})  # 0: neg, 1: neu, 2: pos

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['label'], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("✅ Model trained and saved to models/sentiment_model.pkl")
