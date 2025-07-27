import streamlit as st
import joblib

# Load model and vectorizer
clf, vectorizer = joblib.load("models/model.pkl")

# Set up Streamlit app
st.set_page_config(page_title="Toxic Comment Detector", layout="centered")
st.title("🧠 Toxic Comment Classification")
st.markdown("""
Type or paste a comment below to detect toxic content.  
The model will analyze and flag categories like `insult`, `threat`, `obscene`, and more.
""")

# Input area
with st.form("toxicity_form"):
    comment = st.text_area("💬 Enter a comment:", height=150, placeholder="e.g., You are a complete idiot and a disgrace.")
    submit = st.form_submit_button("Analyze")

# Toxicity Labels
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Inference logic
if submit:
    if comment.strip():
        # Transform input
        X = vectorizer.transform([comment])
        y_pred = clf.predict(X)[0]

        st.subheader("⚠️ Prediction Results:")
        for label, value in zip(labels, y_pred):
            st.markdown(f"- **{label.capitalize()}**: {'✅ Detected' if value else '❌ Clean'}")

        # Optional summary badge
        if any(y_pred):
            st.error("🚨 Toxic content detected in one or more categories.")
        else:
            st.success("✅ This comment appears clean and safe.")

    else:
        st.warning("⚠️ Please enter a comment to analyze.")
