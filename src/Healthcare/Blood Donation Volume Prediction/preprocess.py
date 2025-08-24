# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/blood_donor_dataset.csv"

def preprocess_data():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Features and target
    features = ["months_since_first_donation", "number_of_donation", "city", "blood_group"]
    target = "pints_donated"

    X = df[features]
    y = df[target]

    # Encode categorical columns
    le_city = LabelEncoder()
    le_blood = LabelEncoder()

    X["city"] = le_city.fit_transform(X["city"])
    X["blood_group"] = le_blood.fit_transform(X["blood_group"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    encoders = {
        "city": le_city,
        "blood_group": le_blood,
    }

    return X_train, X_test, y_train, y_test, encoders

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoders = preprocess_data()
    print("✅ Preprocessing complete")
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
