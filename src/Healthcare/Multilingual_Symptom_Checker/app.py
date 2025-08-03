import streamlit as st
from chatbot_utils import translate_to_english, translate_from_english, get_response_from_gemini, clean_ai_response
from config import configure_gemini 

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
}

model = configure_gemini() 

st.set_page_config(page_title="Multilingual Symptom Checker", page_icon=":hospital:", layout="wide")
st.title("Multilingual Symptom Checker :hospital:")

selected_lang = st.selectbox("Select Language", list(LANGUAGES.keys()))
user_input = st.text_area("Describe your symptoms: " , height=150)

if st.button("Check Symptoms"):
    if user_input.strip():
        translated_input = translate_to_english(user_input, LANGUAGES[selected_lang])
        llm_response = get_response_from_gemini(translated_input, model)
        translated_output = translate_from_english(llm_response, LANGUAGES[selected_lang])

        st.markdown("### 💬 Diagnostic Suggestion")
        st.success(translated_output)
    else:
        st.warning("Please enter symptoms.")