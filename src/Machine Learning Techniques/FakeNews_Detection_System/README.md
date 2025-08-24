# 📰 Fake News Detection Web App

A **Streamlit**-based web application that uses a pre-trained **Machine Learning model** to classify news articles as **Real** or **Fake**.

## 📌 Features
- Predicts whether a given news headline or article is **real** or **fake**.
- Uses **TF-IDF Vectorization** for text preprocessing.
- Powered by a pre-trained model (`.pkl` or `.pkl` format).
- Simple, clean, and user-friendly **Streamlit UI**.

## 📂 Project Structure
```bash
📁 FakeNews_Detection_System
├── model.pkl # Trained ML model (can also be in .h5 format)
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── app.py # Streamlit app script
├── requirements.txt # Python dependencies
├── Fake.csv # Fake news dataset
├── True.csv # Real news dataset
└── README.md # Project documentation
```


## 🚀 Installation & Setup

1. **Clone this repository**
```bash
git clone https://github.com/PRIYANSHU2026/DataSentience-AIML.git
cd FakeNews_Detection_System
```

2.Create a virtual environment
```bash
python -m venv myenv
```

3.Activate the virtual environment

Windows:
```bash
myenv\Scripts\activate
```

Linux/Mac:
```bash
source myenv/bin/activate
```

4.Install dependencies
```bash
pip install -r requirements.txt
```

5.Run the Streamlit app
```bash
streamlit run app.py
```

## 🧠 Model Information

Algorithm: Logistic Regression / Naive Bayes (depends on your training)

Text Processing: TF-IDF Vectorization

Dataset: ISOT Fake & Real News Dataset or any custom dataset.

## 📊 Example

Input:
    "Government announces new AI policy to boost startups."

Output:
    ✅ Real News

## 🙋‍♂️ Contributor
👤 Divyanshu Giri  
        GitHub: [Divyanshu-hash](https://github.com/Divyanshu-hash)
        Email: [rishugiri056@gmail.com](rishugiri056@gmail.com)
