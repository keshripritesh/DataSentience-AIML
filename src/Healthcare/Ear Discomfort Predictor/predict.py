import joblib
import pandas as pd

MODEL_PATH = "model/ear_discomfort_predictor.pkl"

def predict(input_data: dict):
    # Load saved model + encoders
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    encoders = saved["encoders"]
    target_encoder = saved["target_encoder"]

    # Convert input into DataFrame
    df = pd.DataFrame([input_data])

    # Encode categorical features
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    # Predict
    pred = model.predict(df)
    return target_encoder.inverse_transform(pred)[0]

if __name__ == "__main__":
    sample = {
        "Perceived_Hearing_Meaning": "Staying independent and alert",
        "Hearing_FOMO": "Sometimes",
        "Hearing_Test_Barrier": "Cost",
        "Missed_Important_Sounds": "Yes, in family conversations",
        "Left_Out_Due_To_Hearing": "Yes, often",
        "Daily_Headphone_Use": "1-2 hours",
        "Belief_Early_Hearing_Care": 5,
        "Last_Hearing_Test_Method": "Self - application",
        "Interest_in_Hearing_App": "Yes",
        "Desired_App_Features": "Privacy",
        "Awareness_on_hearing_and_Willingness_to_invest": "Yes",
        "Paid_App_Test_Interest": "Maybe, if it offers good value",
        "Age_group": "18 - 24"
    }

    print("Prediction:", predict(sample))
