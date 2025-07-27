# preprocess.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def get_preprocessor():
    # Define columns
    categorical_cols = [
        'fuel', 'transmission', 'body', 'Color', 'Engine Type', 'Drive Type',
        'Steering Type', 'owner_type', 'state', 'City', 'Gear Box'
    ]
    numerical_cols = [
        'myear', 'km', 'No of Cylinder', 'Length', 'Width', 'Height', 'Seats'
    ]

    # Pipelines
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numerical_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

    return preprocessor
