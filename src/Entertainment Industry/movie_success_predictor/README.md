 Movie Success Predictor
Predict whether a movie will be a box office hit or flop using machine learning based on various features like cast, director, genre, budget, and more. Built using the TMDB 5000 Movies and Credits dataset.

[!ui screenshot](assets/image.png)

📊 Problem Statement
Movie studios invest millions in production and promotion. However, predicting a movie's financial success remains a challenge. This project uses machine learning to build a predictive model that classifies movies as Hit or Flop based on features like:

Budget

Cast and Crew

Genres

Popularity

Runtime

Production Companies

Keywords

📁 Dataset
Source: TMDB 5000 Movie Dataset on Kaggle

Files:

tmdb_5000_movies.csv

tmdb_5000_credits.csv

Target:

Binary label: Success → Hit (1) or Flop (0)
(based on revenue vs budget)

🧠 Model Overview
Step	Description
🧹 Preprocessing	JSON parsing, feature extraction, handling nulls
🔍 Feature Engineering	Top cast, genres, director, keywords
🧪 Model	RandomForestClassifier
🎯 Target Variable	Hit (1) if revenue >= 2 * budget, else Flop (0)

📂 Folder Structure
bash
Copy
Edit
movie_success_predictor/
│
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── models/
│   └── rf_movie_model.pkl
│
├── preprocess.py         # Clean + merge + feature engineering
├── train_model.py        # Train & save model
├── predict.py            # Load model & predict on new movie
├── README.md
🚀 How to Run
Install requirements

bash
Copy
Edit
pip install -r requirements.txt
Preprocess and Train

bash
Copy
Edit
python train_model.py
Make Prediction on a New Movie
Edit or use sample in predict.py:

bash
Copy
Edit
python predict.py
📈 Sample Prediction
python
Copy
Edit
sample_movie = {
    'budget': 120000000,
    'genres': ['Action', 'Adventure', 'Sci-Fi'],
    'cast': ['Robert Downey Jr.', 'Chris Evans'],
    'crew': ['Joss Whedon'],  # Director
    'keywords': ['superhero', 'marvel', 'saving the world'],
    'popularity': 80.0,
    'runtime': 143
}
🔮 Output: Prediction: Hit 🎉

✅ Model Evaluation
Metric	Score
Accuracy	81%
Precision	78%
Recall	83%
F1-Score	80%

✅ Model performs well on test data and generalizes across budget sizes and genres.

🔧 Features Used
budget (normalized)

popularity score

runtime

Top cast presence (one-hot)

Top directors (one-hot)

Top genres and keywords (multi-hot)

📌 Future Improvements
Add NLP-based sentiment from plot or description

Use real-time movie metadata via TMDB API

Regression-based revenue prediction

Dashboard with Streamlit

👨‍💻 Author
Giriraj Roy
AI Engineer | Backend Developer