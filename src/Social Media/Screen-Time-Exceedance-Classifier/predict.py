# predict.py
import joblib
import numpy as np

def predict_exceedance(age, gender, device, location, ed_rec_ratio):
    model = joblib.load('model/screen_time_classifier.pkl')
    le_gender = joblib.load('model/le_gender.pkl')
    le_device = joblib.load('model/le_device.pkl')
    le_location = joblib.load('model/le_location.pkl')

    gender_enc = le_gender.transform([gender])[0]
    device_enc = le_device.transform([device])[0]
    location_enc = le_location.transform([location])[0]

    input_data = np.array([[age, gender_enc, device_enc, location_enc, ed_rec_ratio]])
    prediction = model.predict(input_data)

    return bool(prediction[0])

# Example usage
if __name__ == "__main__":
    result = predict_exceedance(13, "Male", "Smartphone", "Urban", 0.42)
    print("Exceeded recommended screen time?" , result)
