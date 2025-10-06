import streamlit as st
from datetime import datetime
import os
from mistralai import Mistral
from dotenv import load_dotenv

# ----------------- Load API Key -----------------
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")   # make sure you set this in your .env file
client = Mistral(api_key=api_key) if api_key else None

# ----------------- Page Config -----------------
st.set_page_config(page_title="AI Astrologer", page_icon="🔮", layout="centered")

# ----------------- App Title -----------------
st.title("🔮 AI Astrologer")
st.markdown("Welcome! Enter your details below to get your personalized astrological prediction.")

# ----------------- Input Fields -----------------
name = st.text_input("Your Name 🧑")
dob = st.date_input("Date of Birth 📅", min_value=datetime(1900, 1, 1))

# Time input in 24h → Convert to 12h AM/PM format
tob_24 = st.time_input("Time of Birth ⏰")
tob = tob_24.strftime("%I:%M %p")  # ✅ convert to clock style (AM/PM)

# Show user-friendly formatted time immediately
st.caption(f"🕑 Selected Time: **{tob}**")

place = st.text_input("Place of Birth 🌍")
user_question = st.text_area("Ask your question ❓")

# ----------------- Prediction Button -----------------
if st.button("Get Prediction 🔮"):
    if client:
        response = client.chat.complete(
            model="mistral-tiny",
            messages=[
                {"role": "system", "content": "You are an expert astrologer."},
                {"role": "user", "content": f"Name: {name}, DOB: {dob}, TOB: {tob}, Place: {place}, Question: {user_question}"}
            ]
        )

        # ✅ Fix: use `.content` instead of ["content"]
        st.info(response.choices[0].message.content)

    else:
        st.error("❌ API key not found. Please check your .env file.")

# ----------------- Footer -----------------
st.markdown("---")
st.markdown("<h6 style='text-align: center; color: orange;'>✨ Developed by Pritesh Keshri ✨</h6>", unsafe_allow_html=True)
