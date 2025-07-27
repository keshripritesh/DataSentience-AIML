import streamlit as st
import joblib

# Load model components
clf, vectorizer, le = joblib.load("models/model.pkl")

# App title
st.set_page_config(page_title="MBTI Personality Predictor", layout="centered")
st.title("🧠 MBTI Personality Type Predictor")
st.markdown("Enter a short paragraph, social media post, or text and get your MBTI prediction.")

# Input form
with st.form("mbti_form"):
    user_input = st.text_area("📝 Paste your text here:", height=150)
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted and user_input.strip():
    X = vectorizer.transform([user_input])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    top3 = sorted(zip(le.classes_, proba), key=lambda x: -x[1])[:3]

    st.success(f"🎯 Predicted MBTI Type: `{label}`")
    st.subheader("📊 Top 3 Probabilities:")
    for t, p in top3:
        st.write(f"**{t}** — {round(p * 100, 2)}%")

elif submitted:
    st.warning("⚠️ Please enter some text to analyze.")
