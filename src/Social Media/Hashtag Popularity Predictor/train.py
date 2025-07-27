import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from preprocess import clean_hashtag

# 📥 Load dataset
df = pd.read_csv("data/trending_hashtags.csv")
print("📊 Columns in the dataset:", df.columns.tolist())

# 🧼 Clean hashtag column
df['hashtag'] = df['hashtag'].astype(str).apply(clean_hashtag)

# 🧹 Drop missing values
df = df.dropna(subset=['hashtag', 'mentions', 'estimated_reach', 'sentiment_score'])

# 🎯 Create binary label: 1 if high reach, else 0
threshold = df['estimated_reach'].median()
df['is_trending'] = (df['estimated_reach'] > threshold).astype(int)

# 📦 Feature matrix
X = df[['hashtag', 'mentions', 'sentiment_score']]
y = df['is_trending']

# 🔁 Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('tag_tfidf', TfidfVectorizer(max_features=1000), 'hashtag'),
        ('scaler', StandardScaler(), ['mentions', 'sentiment_score'])
    ]
)

# 🧪 Transform features
X_processed = preprocessor.fit_transform(X)

# ✂️ Split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 🧠 Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📊 Evaluation
y_pred = model.predict(X_test)
print("🧠 Classification Report:")
print(classification_report(y_test, y_pred))

# 💾 Save
os.makedirs("models", exist_ok=True)
with open("models/popularity_model.pkl", "wb") as f:
    pickle.dump((preprocessor, model), f)

print("✅ Model trained and saved at models/popularity_model.pkl")
