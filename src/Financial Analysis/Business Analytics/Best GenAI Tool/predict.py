# predict.py

import pandas as pd
import joblib
from preprocess import preprocess_input

def predict_genai_tool(sample_input):
    model = joblib.load("model/genai_model.pkl")
    input_df = pd.DataFrame([sample_input])
    processed_input = preprocess_input(input_df)
    prediction = model.predict(processed_input)
    return prediction[0]

# Example Usage
if __name__ == "__main__":
    sample = {
        "Industry": "Telecom",
        "Country": "USA",
        "Adoption Year": 2023,
        "Number of Employees Impacted": 5000,
        "New Roles Created": 10,
        "Training Hours Provided": 2000,
        "Productivity Change (%)": 12.5,
        "Employee Sentiment": "AI improved workflow but caused anxiety about job security"
    }

    result = predict_genai_tool(sample)
    print(f"Recommended GenAI Tool: {result}")
