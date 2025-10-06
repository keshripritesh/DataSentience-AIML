# 📰 Fake News Detection Model

This project is a machine learning-based system that detects fake news using Natural Language Processing (NLP) techniques. It classifies news articles as **Fake (1)** or **Real (0)** using a simple yet effective **Logistic Regression** model.

---

## 📚 Dataset

The dataset is sourced from Kaggle:

🔗 [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

It contains the following columns:
- `id`: Unique ID for the news article
- `title`: Title of the article
- `author`: Author of the article
- `text`: Full text of the news article
- `label`: Classification label  
  - `1`: Fake  
  - `0`: Real

---
```bash
Fake-News-Detection/
│
├── 📁 data/                      
│   ├── fake.csv                  # Fake news dataset
│   └── real.csv                  # Real news dataset
│
├── 📁 models/                    
│   ├── vectorizer.joblib         # Saved TF-IDF/CountVectorizer
│   └── model.joblib              # Trained Logistic Regression model
│
├── Fake_News_Detection.ipynb     # Main Jupyter Notebook (Preprocessing + Training + Evaluation)
├── requirements.txt              # Dependencies
└── README.md                     # Project overview
```

## 🔧 Features

- Uploads dataset via Colab
- Handles missing values
- Preprocesses text (lowercasing, removing punctuation, etc.)
- Vectorizes news using **TF-IDF Vectorizer**
- Trains a **Logistic Regression model** for binary classification
- Evaluates using accuracy score, confusion matrix, and classification report

---

## 🤖 Model Used

We used **Logistic Regression**, which is:
- Simple to implement
- Fast to train
- Easy to interpret
- Well-suited for binary classification tasks like detecting fake vs. real news

---

## ⚙️ Technologies Used

- Python
- Google Colab
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Matplotlib / Seaborn (optional for visualization)

---
## Usage
-You can test the trained Logistic Regression model using the datasets provided (fake.csv and real.csv) in the data/ folder.