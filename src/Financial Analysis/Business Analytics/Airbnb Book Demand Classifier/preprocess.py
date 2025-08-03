# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Drop rows with missing reviews_per_month (since it's our label basis)
    df = df.dropna(subset=['reviews_per_month'])

    # Create binary target
    df['is_frequently_booked'] = (df['reviews_per_month'] > 1).astype(int)

    # Features to use
    features = [
        'neighbourhood_group', 'room_type', 'price',
        'minimum_nights', 'number_of_reviews',
        'availability_365', 'calculated_host_listings_count'
    ]
    target = 'is_frequently_booked'

    X = df[features]
    y = df[target]

    # Identify categorical and numerical features
    cat_features = ['neighbourhood_group', 'room_type']
    num_features = [col for col in X.columns if col not in cat_features]

    # Pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Full preprocessing
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_features),
        ('cat', categorical_pipeline, cat_features)
    ])

    return X, y, preprocessor
