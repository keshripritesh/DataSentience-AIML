# preprocess.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def load_and_preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Remove listings with invalid prices
    df = df[(df['price'] > 0) & (df['price'] <= 1000)]

    # Features to use
    features = [
        'neighbourhood_group', 'room_type',
        'latitude', 'longitude',
        'minimum_nights', 'availability_365',
        'number_of_reviews', 'reviews_per_month',
        'calculated_host_listings_count'
    ]
    target = 'price'

    # Drop rows with missing values in required columns
    df = df.dropna(subset=features)

    # Log transform the target for skew handling
    df['log_price'] = np.log1p(df['price'])

    X = df[features]
    y = df['log_price']

    # Categorical and numerical features
    cat_features = ['neighbourhood_group', 'room_type']
    num_features = [col for col in features if col not in cat_features]

    # Pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_features),
        ('cat', categorical_pipeline, cat_features)
    ])

    return X, y, preprocessor
