# 🎬 Anime Recommender System (Content-Based Filtering)

## 📌 Overview

This project implements a **Content-Based Recommender System** for anime using **Natural Language Processing (NLP)** techniques. Instead of relying on user ratings or collaborative filtering, the system recommends anime by analyzing their **descriptions, genres, and metadata** to find similar titles.

---

## ⚙️ Workflow

1. **Data Loading**

   * Dataset: *MyAnimeList 2023* from Kaggle.
   * Important features: `Name`, `Genres`, `Synopsis`, `Type`, `Source`, `Studios`, `Producers`, `Popularity`, `Favorites`, `Episodes`, `Rating`, `Rank`.

2. **Data Preprocessing**

   * Missing values handled.
   * Text normalization steps applied: stopword removal, tokenization, and lemmatization.
   * Combined multiple columns into a unified **`tags`** column to represent each anime.

3. **Feature Extraction**

   * Used **CountVectorizer** to convert the `tags` column into numerical vectors.
   * Representation limited to top features (Bag of Words model).

4. **Similarity Computation**

   * Used **Cosine Similarity** to compute closeness between anime vectors.
   * Constructed a similarity matrix for recommendations.

5. **Recommendation Function**

   * Input: Anime title.
   * Output: Top N most similar anime titles.

---

## 🧠 Algorithms & Concepts Used

### 🔹 Stopwords

* **Definition:** Common words (like *is, the, an, in*) that add little meaning in text analysis.
* **Why removed?** They don’t help in identifying content similarity.

### 🔹 Tokenization

* **Definition:** Breaking text into smaller units (words or tokens).
* Example: *"Naruto is awesome"* → `["Naruto", "is", "awesome"]`.

### 🔹 Lemmatization

* **Definition:** Reducing words to their root form based on meaning.
* Example: *"running", "runs" → "run"*.
* **Why?** Ensures similar words are treated as the same concept.

### 🔹 CountVectorizer

* **Definition:** A feature extraction method from scikit-learn. Converts text into a **matrix of token counts** (Bag of Words model).
* **Example:**

  * Sentences: \["I love anime", "I hate anime"]
  * Vocab: `[I, love, hate, anime]`
  * Matrix:

    ```
    I     love   hate   anime
    [1,    1,     0,     1]
    [1,    0,     1,     1]
    ```

### 🔹 Cosine Similarity

* **Definition:** A metric that measures similarity between two vectors based on the cosine of the angle between them.

* **Why?** Independent of vector length → useful for comparing text documents.

* **Formula:**

  ```
  cos(θ) = (A · B) / (||A|| ||B||)
  ```

  where A and B are feature vectors.

* **Interpretation:**

  * `1` → identical vectors.
  * `0` → no similarity.
  * `-1` → opposite vectors.

---

## 🚀 Future Improvements

* Replace Bag of Words with **TF-IDF** or **CBOW** for better feature weighting.
* Use **Word2Vec / Transformer embeddings** for semantic similarity.
* Implement **hybrid filtering** (content + user ratings).
* A Streamlit Deployable Website

---
