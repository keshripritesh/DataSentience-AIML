import streamlit as st
import pickle
import re

# -------------------------
# Load the trained model & vectorizer
# -------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# -------------------------
# Preprocessing function
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# -------------------------
# Streamlit App
# -------------------------
st.title("📰 Fake News Detection App")
st.write("Enter a news article below, and the model will predict whether it's **Fake** or **True**.")

# Text input
user_input = st.text_area("Paste your news article here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        # Preprocess and vectorize
        processed_text = preprocess_text(user_input)
        vectorized_text = tfidf_vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(vectorized_text)[0]
        
        # Display result
        if prediction == 1:
            st.error("🚨 This news is predicted to be **FAKE**.")
        else:
            st.success("✅ This news is predicted to be **TRUE**.")

st.markdown("---")
