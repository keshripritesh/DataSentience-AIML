# 🔥 Hashtag Popularity Predictor

Predict whether a hashtag is likely to trend on social media using its metadata like mentions and sentiment score.

---

[!ui screenshot](assets/image.png)

## 📌 Project Description

Hashtags are powerful tools in digital marketing and social communication. This machine learning project predicts whether a hashtag is likely to trend (i.e., become popular) based on:

- Its **text**
- The number of **mentions**
- Its **sentiment score**

This helps marketers, influencers, and trend analysts understand what kind of tags are likely to gain momentum.

---

## 📊 Dataset

**Dataset Used:**  
[Trending Hashtags Dataset (custom)](https://www.kaggle.com/) *(uploaded manually)*

### 📁 Columns:
| Column Name        | Description                                  |
|--------------------|----------------------------------------------|
| `date`             | Date when the hashtag data was collected     |
| `hashtag`          | The hashtag text (e.g., #WorldCup2022)       |
| `mentions`         | Number of times the hashtag was used         |
| `estimated_reach`  | Estimated number of users reached            |
| `sentiment_score`  | Sentiment polarity score (-1 to 1)           |
| `top_country`      | Country where hashtag trended the most       |

Since the dataset doesn't have a direct "is trending?" column, we **generate a binary label**:

is_trending = 1 if estimated_reach > median(estimated_reach) else 0

yaml
Copy
Edit

---

## 💡 Features Used

- `hashtag` → Text vectorized using TF-IDF
- `mentions` → Count of hashtag mentions
- `sentiment_score` → Pre-computed polarity from tweets

---

## 🚀 Model

| Component            | Details                          |
|----------------------|----------------------------------|
| Preprocessing        | TF-IDF + StandardScaler          |
| Model Used           | RandomForestClassifier           |
| Training/Test Split  | 80/20                            |
| Saved Model          | `models/popularity_model.pkl`    |

---

## 🧠 How It Works

- Preprocess the text using `preprocess.py`
- Extract features from `hashtag`, `mentions`, `sentiment_score`
- Train a `RandomForestClassifier` to predict `is_trending`
- Save the trained model
- Use `predict.py` to make predictions from new hashtag input

---

## 📁 Project Structure

Hashtag Popularity Predictor/
├── data/
│ └── trending_hashtags.csv
├── models/
│ └── popularity_model.pkl
├── preprocess.py
├── train.py
├── predict.py
└── README.md

yaml
Copy
Edit

---

## ⚙️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/hashtag-popularity-predictor.git
cd hashtag-popularity-predictor
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Train the model
bash
Copy
Edit
python train.py
4. Predict popularity of a new hashtag
bash
Copy
Edit
python predict.py