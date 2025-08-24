from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# === Load Model and Encoders ===
model = joblib.load(open("Random_Forest_best_model.pkl", "rb"))
encoders = joblib.load(open("encoders.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # === Behavioral Questions (A1–A10) ===
        A1 = 1 if request.form["A1"] == "yes" else 0
        A2 = 1 if request.form["A2"] == "yes" else 0
        A3 = 1 if request.form["A3"] == "yes" else 0
        A4 = 1 if request.form["A4"] == "yes" else 0
        A5 = 1 if request.form["A5"] == "yes" else 0
        A6 = 1 if request.form["A6"] == "yes" else 0
        A7 = 1 if request.form["A7"] == "yes" else 0
        A8 = 1 if request.form["A8"] == "yes" else 0
        A9 = 1 if request.form["A9"] == "yes" else 0
        A10 = 1 if request.form["A10"] == "yes" else 0

        # === Demographic Data ===
        age = int(request.form["age"])
        gender = encoders["gender"].transform([request.form["gender"]])[0]
        ethnicity = encoders["ethnicity"].transform([request.form["ethnicity"]])[0]
        jaundice = encoders["jaundice"].transform([request.form["jaundice"]])[0]
        austim = encoders["austim"].transform([request.form["austim"]])[0]
        country = encoders["contry_of_res"].transform([request.form["contry_of_res"]])[0]  # Fix typo if needed

        # === Feature Vector === (must match model training order)
        features = np.array([[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,
                              age, gender, ethnicity, jaundice, austim, country]])

        # === Predict ===
        prediction = model.predict(features)[0]
        result = "High Risk of Autism" if prediction == 1 else "Low Risk of Autism"

        return render_template("result.html", result=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
