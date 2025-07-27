import pandas as pd
import joblib

# Load model
model = joblib.load("models/movie_success_model.pkl")

# Sample movie input
new_movie = {
    'budget': 100000000,
    'popularity': 25.0,
    'runtime': 130,
    'genres': 'Action',
    'cast': 'Robert Downey Jr.',
    'director': 'Jon Favreau'
}

def predict_new_movie(movie_dict):
    df = pd.DataFrame([movie_dict])
    prediction = model.predict(df)
    return "Hit" if prediction[0] else "Flop"

print(predict_new_movie(new_movie))
