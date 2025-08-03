import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from preprocess import clean_and_encode

def train(data_path='data/interview_data.csv'):
    print("📥 Loading data...")
    df = pd.read_csv(data_path)

    print("🧼 Cleaning and encoding...")
    df_clean = clean_and_encode(df)

    # Target column
    target_col = 'Whether joined the company or not'

    if target_col not in df_clean.columns:
        raise KeyError(f"❌ Target column '{target_col}' not found in preprocessed data.")

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    print("🧪 Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("🎯 Training Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    print("💾 Saving model, scaler, and column list...")
    with open('models/candidate_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('models/train_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    print("📊 Classification Report:")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    print("✅ Training complete.")

if __name__ == '__main__':
    train()
