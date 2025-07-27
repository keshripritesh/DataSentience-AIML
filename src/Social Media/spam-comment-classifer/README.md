# 📺 YouTube Comment Spam Classifier

A machine learning project that detects whether a YouTube comment is **spam** or **not spam** using text classification techniques. Built using the full [YouTube Spam Collection dataset](https://www.kaggle.com/datasets/raviram544/youtube-spam-comment-dataset), this project helps in understanding NLP preprocessing, vectorization, and spam detection using classical ML algorithms like **Multinomial Naive Bayes**.

---

[!ui screenshot](assets/image.png)

## 🧠 Project Motivation

Spam comments are a major issue on platforms like YouTube. They degrade user experience, spread misinformation, and clutter content. This classifier provides a lightweight solution to automatically flag spammy content based on comment text alone.

---

## 📂 Folder Structure

YouTube Comment Spam Classifier/
├── data/
│ ├── Youtube01-Psy.csv
│ ├── Youtube02-KatyPerry.csv
│ ├── Youtube03-LMFAO.csv
│ ├── Youtube04-Eminem.csv
│ └── Youtube05-Shakira.csv
├── models/
│ └── spam_model.pkl
├── train.py
├── predict.py
├── preprocess.py
└── README.md

yaml
Copy
Edit

---

## 📊 Dataset

This project uses **all 5 subsets** of the YouTube Spam Collection Dataset:

- Source: [Kaggle - YouTube Spam Dataset](https://www.kaggle.com/datasets/raviram544/youtube-spam-comment-dataset)
- Collected from videos by:
  - Psy
  - Katy Perry
  - LMFAO
  - Eminem
  - Shakira
- Fields:
  - `CONTENT`: the comment text
  - `CLASS`: `1` = spam, `0` = not spam

---

## ⚙️ How It Works

### 1. Preprocessing
Each comment is:
- Lowercased
- URLs are replaced with `url`
- Punctuation is removed
- Extra whitespaces are stripped

### 2. Vectorization
Text is converted into numeric features using **TF-IDF vectorizer** with a limit of 5,000 features.

### 3. Model Training
We use **Multinomial Naive Bayes**, a popular algorithm for text classification and spam detection.

### 4. Prediction
The model predicts a label (`spam` or `not spam`) based on new input comments from users.

---

## 🚀 How to Run

### 📦 1. Install Requirements
No `requirements.txt` needed — uses basic packages.

```bash
pip install pandas scikit-learn
🛠 2. Train the Model
bash
Copy
Edit
python train.py
Loads all 5 CSVs from data/

Preprocesses comments

Trains the model

Saves model as models/spam_model.pkl

🧪 3. Predict on New Comments
bash
Copy
Edit
python predict.py
Example:

mathematica
Copy
Edit
Enter a YouTube comment: Get 10,000 subs instantly! Visit bit.ly/freeboost 🚀
Prediction: Spam

Enter a YouTube comment: I really enjoyed this song, brings back memories!
Prediction: Not Spam
🧼 Sample Preprocessing
python
Copy
Edit
from preprocess import clean_text

print(clean_text("Check out my channel!!! http://bit.ly/spam-link"))
# Output: "check out my channel url"
🧠 Why Multinomial Naive Bayes?
Ideal for word-based classification

Very fast and lightweight

Performs better than Logistic Regression on small to medium-sized text datasets for spam detection