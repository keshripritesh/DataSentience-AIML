# Comment Classification – Psychotic Depression Detection

This module provides an implementation for classifying **social media comments** as related to **psychotic depression** or not.  
It demonstrates the application of machine learning techniques for text-based mental health analysis.

---

## 📂 Files
- `comments_classification.ipynb` – Jupyter Notebook with preprocessing, training, and evaluation steps.
- `README.md` – documentation and usage guide.

---

## 🚀 How to Use
1. Navigate into this folder:
   ```bash
   cd "src/Social Media/CommentClassification"
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook comments_classification.ipynb
   ```

3. Run the notebook step by step to:
   - Preprocess and clean text data
   - Extract features (e.g., TF-IDF or embeddings)
   - Train the classification model
   - Evaluate performance on sample data

---

## 🛠 Requirements
Install dependencies before running the notebook:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

*(You can also generate a full `requirements.txt` by running `pip freeze > requirements.txt` in your environment.)*

---

## 📊 Output
The notebook outputs evaluation metrics such as:
- Accuracy
- Precision
- Recall
- F1-score

---

## 🌱 Future Work
- Add deep learning–based models (LSTM, transformers, etc.)
- Expand dataset with more annotated comments
- Provide pre-trained model weights for reuse

---

## 🤝 Contribution
Contributions are welcome! Please submit an issue or open a pull request if you’d like to extend or improve this module.
