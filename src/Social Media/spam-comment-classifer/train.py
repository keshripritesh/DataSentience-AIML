import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle
from preprocess import clean_text

# Load all data files
data_dir = "data"
files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in files]
df = pd.concat(dfs, ignore_index=True)

# Clean and prepare
df['cleaned'] = df['CONTENT'].astype(str).apply(clean_text)
X = df['cleaned']
y = df['CLASS']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model
os.makedirs("models", exist_ok=True)
with open("models/spam_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("✅ Trained and saved spam_model.pkl")
