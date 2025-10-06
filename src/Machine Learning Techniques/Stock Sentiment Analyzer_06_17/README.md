# 📈 Stock Sentiment Analyzer

This tool analyzes the **public sentiment** around a stock based on recent **news articles**. It helps investors and traders assess market mood and predict possible market movement direction.

---
## 🔍 UI Preview

Here is a preview of the UI:
![UI Screenshot](assets/1.jpeg)
![UI Screenshot](assets/2.jpeg)
## 🚀 Features

- 🔍 **Input Stock Ticker** (e.g., `TCS`, `INFY`, etc.)
- 📰 **Scrape Latest News Headlines** using NewsAPI
- 🤖 **Sentiment Analysis** using `cardiffnlp/twitter-roberta-base-sentiment` (a state-of-the-art transformer model)
- 📊 **Output Sentiment Scores**:
  - % Positive
  - % Neutral
  - % Negative
- 📈 **Market Impact Prediction** based on overall sentiment polarity

---

## 🧠 How It Works

1. **Input**: User enters a stock name or ticker.
2. **Scraping**: Tool fetches recent news headlines related to that stock.
3. **Analysis**: Each headline is run through a pretrained Roberta model to classify its sentiment.
4. **Result**:
   - Overall sentiment distribution (%)
   - Predicted impact on stock sentiment (Bullish / Bearish / Neutral)

---

## 🗂️ File Structure

Financial AI/
└── stock-sentiment-analyser/
├── app.py # Main application entry point
├── news_scraper.py # Scrapes latest news using NewsAPI
├── sentiment_model.py # Loads transformer model and analyzes sentiment
└── utils.py # Contains helper function to predict market impact

yaml
Copy
Edit

---

## 📦 Requirements

Make sure to install dependencies:

```bash
pip install transformers torch requests