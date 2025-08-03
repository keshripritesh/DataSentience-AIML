#!/usr/bin/env python3
"""
Fixed training script for Emotion Recognition Model
Generates the required pickle files for the Flask app
"""

import pandas as pd
import numpy as np
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

print("🤖 Starting Emotion Recognition Model Training...")

# Reading datasets with proper paths
try:
    train = pd.read_csv("Dataset/train.txt", delimiter=';', header=None, names=['sentence','label'])
    test = pd.read_csv("Dataset/test.txt", delimiter=';', header=None, names=['sentence','label'])
    val = pd.read_csv("Dataset/val.txt", delimiter=';', header=None, names=['sentence','label'])
    
    print(f"✅ Data loaded successfully!")
    print(f"   📊 Train: {len(train)} samples")
    print(f"   📊 Test: {len(test)} samples") 
    print(f"   📊 Validation: {len(val)} samples")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Combine all data
df_data = pd.concat([train, test, val], ignore_index=True)
print(f"📈 Total dataset size: {len(df_data)} samples")

# Show emotion distribution
emotion_counts = df_data['label'].value_counts()
print(f"🎭 Emotion distribution:")
for emotion, count in emotion_counts.items():
    print(f"   {emotion}: {count}")

# Text preprocessing
print("\n🔄 Preprocessing text data...")
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english', ngram_range=(1,1), tokenizer=token.tokenize)
text = cv.fit_transform(df_data['sentence'])

print(f"✅ Text vectorized: {text.shape[0]} samples, {text.shape[1]} features")

# Save the vectorizer
pickle.dump(cv, open('transform.pkl', 'wb'))
print("💾 Saved transform.pkl")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    text, df_data['label'], test_size=0.30, random_state=5
)

print(f"🔀 Data split:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing: {X_test.shape[0]} samples")

# Train the model
print("\n🧠 Training Multinomial Naive Bayes model...")
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions
predicted = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predicted)

print(f"🎯 Model trained successfully!")
print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Test with sample data
test_data = ['I love the way website looks', 'I am very sad today', 'This makes me angry']
test_results = classifier.predict(cv.transform(test_data))

print(f"\n🧪 Sample predictions:")
for text, emotion in zip(test_data, test_results):
    print(f"   '{text}' → {emotion}")

# Save the model
pickle.dump(classifier, open('nlp.pkl', 'wb'))
print("💾 Saved nlp.pkl")

print(f"\n✅ Training complete! Files created:")
print(f"   📄 nlp.pkl - Trained model")
print(f"   📄 transform.pkl - Text vectorizer")
print(f"\n🚀 Ready to run Flask app with: python app.py")
