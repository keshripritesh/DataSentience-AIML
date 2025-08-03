# preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Encode categorical variables
    le_gender = LabelEncoder()
    le_device = LabelEncoder()
    le_location = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Primary_Device'] = le_device.fit_transform(df['Primary_Device'])
    df['Urban_or_Rural'] = le_location.fit_transform(df['Urban_or_Rural'])

    X = df[['Age', 'Gender', 'Primary_Device', 'Urban_or_Rural', 'Educational_to_Recreational_Ratio']]
    y = df['Exceeded_Recommended_Limit'].astype(int)

    return X, y, le_gender, le_device, le_location
