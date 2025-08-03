import streamlit as st
import google.generativeai as genai

def configure_gemini():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except KeyError:
        st.error("Gemini API key not found in Streamlit secrets.")
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
    return None