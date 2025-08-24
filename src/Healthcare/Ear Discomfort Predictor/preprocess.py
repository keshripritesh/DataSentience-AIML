import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/Hearing well-being Survey Report.csv"

def preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Drop rows with missing values
    df = df.dropna()

    # Target variable
    y = df["Ear_Discomfort_After_Use"]

    # Features (exclude target)
    X = df.drop(columns=["Ear_Discomfort_After_Use"])

    # Encode categorical features
    encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, encoders, target_encoder

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _, _ = preprocess_data()
    print("Training set:", X_train.shape, "Test set:", X_test.shape)
