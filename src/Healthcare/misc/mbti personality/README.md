MBTI Personality Type Classifier
This project presents an NLP-based text classification model designed to predict a user's Myers–Briggs Type Indicator (MBTI) personality type from their written text posts.

📌 Overview
Leveraging the MBTI 16 Personality Dataset from Kaggle, this multi-class classification system employs TF-IDF for text vectorization and Logistic Regression for prediction. The goal is to classify input text into one of the 16 distinct MBTI types (e.g., INFP, ESTJ, etc.).


[!ui screenshot](assets/Screenshot_7-7-2025_21646_localhost.jpeg)


🧾 Dataset
Source: Kaggle - MBTI Personality Type Dataset (or provide the direct URL if you have it)

Key Columns:

type: The assigned MBTI personality type.

posts: A concatenated string of text posts from the user.

🛠️ Project Structure
mbti_personality_classifier/
├── data/
│   └── mbti.csv                   # Raw dataset
├── models/
│   └── trained_model_artifacts.pkl # Serialized model, TF-IDF vectorizer, and label encoder
├── src/
│   ├── preprocess.py              # Data loading and text preprocessing
│   ├── train.py                   # Model training and saving logic
│   └── predict.py                 # Inference script for new text inputs
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation (this file)
🧪 Model Details
Vectorizer: TF-IDF (Term Frequency-Inverse Document Frequency)

max_features: 5000

stop_words: English (removed)

Classifier: Logistic Regression (configured for multi-class classification)

