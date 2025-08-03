#!/usr/bin/env python3
"""
Smart Emotion Recognition with Negation Handling
Uses rule-based negation detection combined with ML model
"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

class SmartEmotionClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.emotion_opposites = {
            'joy': 'sadness',
            'love': 'anger', 
            'surprise': 'fear',
            'sadness': 'joy',
            'anger': 'love',
            'fear': 'surprise'
        }
        
    def detect_negation(self, text):
        """
        Detect if text contains negation patterns
        """
        negation_patterns = [
            r'\b(not|don\'t|doesn\'t|didn\'t|won\'t|can\'t|isn\'t|aren\'t|wasn\'t|weren\'t)\b',
            r'\b(never|no|nobody|nothing|nowhere|neither|nor)\b',
            r'\b(hardly|barely|scarcely)\b'
        ]
        
        text_lower = text.lower()
        for pattern in negation_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        text = text.lower()
        # Keep negations as important markers
        text = re.sub(r"don't", "do_not", text)
        text = re.sub(r"doesn't", "does_not", text)  
        text = re.sub(r"didn't", "did_not", text)
        text = re.sub(r"won't", "will_not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"isn't", "is_not", text)
        text = re.sub(r"aren't", "are_not", text)
        return text
    
    def train(self, sentences, labels):
        """Train the smart emotion classifier"""
        print("🧠 Training Smart Emotion Classifier...")
        
        # Preprocess sentences
        processed_sentences = [self.preprocess_text(sent) for sent in sentences]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=8000,
            min_df=2,
            max_df=0.95
        )
        
        X = self.vectorizer.fit_transform(processed_sentences)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(X, labels)
        
        print("✅ Training completed!")
        
    def predict_emotion(self, text):
        """
        Predict emotion with smart negation handling
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Model not trained yet!")
            
        # Check for negation
        has_negation = self.detect_negation(text)
        
        # Preprocess and vectorize
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        
        # Get prediction and probabilities
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        # Get all emotion probabilities
        emotion_probs = {}
        for i, emotion in enumerate(self.model.classes_):
            emotion_probs[emotion] = probabilities[i] * 100
        
        # Apply negation logic for specific cases
        if has_negation:
            # Check if the predicted emotion makes sense with negation
            negative_indicators = ['good', 'great', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'enjoy']
            positive_indicators = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'scared']
            
            text_lower = text.lower()
            
            # If negation + positive word -> likely negative emotion
            if any(word in text_lower for word in negative_indicators):
                if prediction in ['joy', 'love', 'surprise']:
                    # Override with more appropriate negative emotion
                    if 'feel' in text_lower or 'good' in text_lower:
                        prediction = 'sadness'
                    elif 'like' in text_lower or 'love' in text_lower:
                        prediction = 'anger'
                    
                    confidence = 0.85  # High confidence in negation override
                    
                    # Adjust probabilities
                    emotion_probs = {emotion: 10.0 for emotion in emotion_probs}
                    emotion_probs[prediction] = 85.0
        
        return prediction, confidence * 100, emotion_probs
    
    def save(self, model_path, vectorizer_path):
        """Save the trained model and vectorizer"""
        pickle.dump(self.model, open(model_path, 'wb'))
        pickle.dump(self.vectorizer, open(vectorizer_path, 'wb'))
        print(f"💾 Saved model to {model_path} and vectorizer to {vectorizer_path}")

def main():
    print("🤖 Starting Smart Emotion Recognition Training...")
    
    # Load datasets
    try:
        train = pd.read_csv("Dataset/train.txt", delimiter=';', header=None, names=['sentence','label'])
        test = pd.read_csv("Dataset/test.txt", delimiter=';', header=None, names=['sentence','label'])
        val = pd.read_csv("Dataset/val.txt", delimiter=';', header=None, names=['sentence','label'])
        print("✅ Successfully loaded all datasets")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # Combine data
    df_data = pd.concat([train, test, val], ignore_index=True)
    print(f"📈 Total dataset size: {len(df_data)} samples")
    
    # Create and train smart classifier
    classifier = SmartEmotionClassifier()
    classifier.train(df_data['sentence'].tolist(), df_data['label'].tolist())
    
    # Save the smart model
    classifier.save('nlp_smart.pkl', 'transform_smart.pkl')
    
    # Test on problematic examples
    print("\n🧪 Testing Smart Classifier on problematic examples:")
    test_phrases = [
        "I don't feel so good",
        "I'm not feeling well", 
        "I feel terrible",
        "I'm really sad",
        "This is not good",
        "I don't like this",
        "I'm so happy",
        "This is amazing",
        "I love this",
        "I hate this"
    ]
    
    for phrase in test_phrases:
        emotion, confidence, probs = classifier.predict_emotion(phrase)
        print(f"Text: '{phrase}' -> {emotion} ({confidence:.1f}%)")
    
    print(f"\n✅ Smart emotion classifier training complete!")

if __name__ == "__main__":
    main()
