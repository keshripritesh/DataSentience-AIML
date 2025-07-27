# 🐦 Twitter Sentiment Analyzer

The **Twitter Sentiment Analyzer** is a machine learning project that classifies tweets as **Positive**, **Neutral**, or **Negative**. It uses a preprocessed version of the **Sentiment140 dataset**, which contains 1.6 million tweets labeled for sentiment.

[!ui screenshot](assets/image.png)

The model is built using classic NLP techniques (TF-IDF + Logistic Regression), making it fast, interpretable, and easy to train. This project is suitable for beginners and intermediate ML practitioners and is structured following open-source contribution best practices for programs like **Social Summer of Code (SSoC)**.

---

## 📌 Features

- ✅ Cleans and preprocesses real tweets using regex
- ✅ Uses **TF-IDF** for text vectorization
- ✅ Trains a **Logistic Regression classifier** on Sentiment140
- ✅ CLI-based prediction system via `predict.py`
- ✅ Lightweight and easy to deploy locally
- ✅ Modular file structure for ease of understanding and extension

---

## 📁 Project Structure

Twitter Sentiment Analyzer/
├── data/
│ ├── training.1600000.processed.noemoticon.csv # Full dataset (DO NOT push to GitHub)
│ └── sample_tweets.csv # Mini sample for testing
│
├── models/
│ └── sentiment_model.pkl # Saved model (vectorizer + classifier)
│
├── train.py # Trains the model using Sentiment140
├── predict.py # CLI interface to predict sentiment of input tweet
├── preprocess.py # Utility functions for cleaning tweets
├── README.md

yaml
Copy
Edit

---

## 🧠 Model Details

| Component       | Technique/Model                            |
|----------------|---------------------------------------------|
| Text Cleaning   | Regex-based token cleanup                  |
| Vectorization   | TF-IDF (5000 max features)                  |
| Classifier      | Logistic Regression (`scikit-learn`)        |
| Input           | Raw tweets                                 |
| Output          | Sentiment class: Positive / Neutral / Negative |

---

## 🧪 Dataset

- **Name**: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size**: 1.6 million tweets
- **Labels**:  
  - `0` → Negative  
  - `2` → Neutral  
  - `4` → Positive  

These labels are mapped to:  
- `0 → Negative`  
- `1 → Neutral`  
- `2 → Positive` in our model pipeline.

> 💡 **Note**: Do **not push** the full CSV to GitHub. Instead, use `sample_tweets.csv` or download manually from Kaggle and place it in the `data/` folder.

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/Twitter-Sentiment-Analyzer.git
   cd Twitter-Sentiment-Analyzer
(Optional) Create virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # on Linux/Mac
.\venv\Scripts\activate    # on Windows
Install required packages:

bash
Copy
Edit
pip install scikit-learn pandas
🚀 How to Use
🏋️‍♂️ Step 1: Train the Model
bash
Copy
Edit
python train.py
This will:

Load and clean tweets

Vectorize them using TF-IDF

Train a Logistic Regression model

Save the model as models/sentiment_model.pkl

🔮 Step 2: Predict Tweet Sentiment
bash
Copy
Edit
python predict.py
Example:

mathematica
Copy
Edit
Enter tweet: I absolutely love the new update!
Predicted Sentiment: Positive
📊 Sample Predictions from Dataset
bash
Copy
Edit
Enter tweet: This app is horrible and crashes all the time.
Predicted Sentiment: Negative

Enter tweet: It’s okay, not great but not bad either.
Predicted Sentiment: Neutral

Enter tweet: Love the interface! So sleek and modern.
Predicted Sentiment: Positive
📥 Dataset Download Instructions
If you want to use the full 1.6M tweet dataset, download it from Kaggle:

Sentiment140 on Kaggle

After downloading, place the file as:

bash
Copy
Edit
data/training.1600000.processed.noemoticon.csv
⚠️ DO NOT PUSH this large file to GitHub!