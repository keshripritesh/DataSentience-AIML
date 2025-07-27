import pandas as pd
import ast

def load_and_merge_data(movies_path, credits_path):
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    credits.rename(columns={'movie_id': 'id'}, inplace=True)
    df = movies.merge(credits, on='id')
    return df

def preprocess_data(df):
    # Drop irrelevant columns
    df = df[['budget', 'popularity', 'runtime', 'genres', 'cast', 'crew', 'revenue']]

    # Parse genres and extract first genre name
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x)[0]['name'] if len(ast.literal_eval(x)) > 0 else 'Unknown')

    # Parse cast: take top 3 cast names
    df['cast'] = df['cast'].apply(lambda x: ', '.join([i['name'] for i in ast.literal_eval(x)[:3]]) if len(ast.literal_eval(x)) > 0 else 'Unknown')

    # Parse crew: extract director name
    def get_director(crew_list):
        for member in ast.literal_eval(crew_list):
            if member['job'] == 'Director':
                return member['name']
        return 'Unknown'

    df['director'] = df['crew'].apply(get_director)

    # Drop original crew
    df.drop(columns=['crew'], inplace=True)

    # Label creation: Hit = revenue > budget * 1.5
    df['success'] = df.apply(lambda x: 1 if x['revenue'] > 1.5 * x['budget'] else 0, axis=1)

    # Drop revenue column to avoid leakage
    df.drop(columns=['revenue'], inplace=True)

    # Fill missing values
    df.fillna(0, inplace=True)

    return df

def encode_features(df):
    df_encoded = pd.get_dummies(df, columns=['genres', 'cast', 'director'], drop_first=True)
    return df_encoded

def prepare_data():
    df = load_and_merge_data('data/tmdb_5000_movies.csv', 'data/tmdb_5000_credits.csv')
    df = preprocess_data(df)
    df_encoded = encode_features(df)

    X = df_encoded.drop('success', axis=1)
    y = df_encoded['success']
    return X, y
