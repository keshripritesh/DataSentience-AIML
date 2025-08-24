# 🎬 Movie Recommendation System

This is a **Content-Based Movie Recommendation System** built with Python.  
It suggests movies based on textual similarity of their metadata (overview, genres etc).  

---

## 🚀 Features
- Case-insensitive movie search  
- Text vectorization using **TF-IDF**.  
- Movie similarity measured with **Cosine Similarity**.  
- Top-K recommendations generated for any input movie from the dataset.  
- Evaluation metrics included:
  - ✅ Mean Max Similarity (MMS)  
  - ✅ Intra-List Diversity (ILD@K)  
  - ✅ Coverage@K  


---

## 📂 Project Structure
```
Movie-Recommendation-System/
│
├── data/
│   └── dataset.csv                # Dataset file
│
├── notebook/
│   └── movie_recommended_system.ipynb   # Main notebook
│
├── requirements.txt              # Dependencies
├── README.md                     # Documentation

```

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/PRIYANSHU2026/DataSentience-AIML.git
cd "src/Entertainment Industry/Movie Recommended System"
pip install -r requirements.txt
```

---

## ▶️ Usage
1. Open the Notebook
```bash
jupyter notebook movie_recommended_system.ipynb
```

2. Run all cells
3. To get Recommendations run like below

```python
recommend("Inception")
```

