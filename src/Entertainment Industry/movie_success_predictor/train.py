import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load datasets
movies_df = pd.read_csv("data/tmdb_5000_movies.csv")
credits_df = pd.read_csv("data/tmdb_5000_credits.csv")

# Merge on title
df = movies_df.merge(credits_df, on='title')

# Target: classify hit/flop based on revenue
df = df[df['budget'] != 0]
df['success'] = df['revenue'] > df['budget'] * 2

# Feature selection
df['cast'] = df['cast'].apply(lambda x: eval(x)[0]['name'] if len(eval(x)) > 0 else 'Unknown')
df['director'] = df['crew'].apply(lambda x: next((i['name'] for i in eval(x) if i['job'] == 'Director'), 'Unknown'))

features = df[['budget', 'popularity', 'runtime', 'genres', 'cast', 'director']]
features['genres'] = features['genres'].apply(lambda x: eval(x)[0]['name'] if len(eval(x)) > 0 else 'Unknown')

X = features
y = df['success']

# Preprocessing
numeric_features = ['budget', 'popularity', 'runtime']
categorical_features = ['genres', 'cast', 'director']

numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, 'models/movie_success_model.pkl')
print("Model trained and saved.")
