import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocess import load_data, preprocess

# Load data
X_raw, y = load_data("data/train.csv")
X, vectorizer = preprocess(X_raw, fit=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Train classifier
clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=y.columns))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump((clf, vectorizer), "models/model.pkl")
print("✅ Model saved to models/model.pkl")
