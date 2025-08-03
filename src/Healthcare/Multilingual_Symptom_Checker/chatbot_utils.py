from deep_translator import GoogleTranslator
import google.generativeai as genai
import streamlit as st
import re

def translate_to_english(text , src_lang):
    if src_lang.lower()=="english":
        return text
    return GoogleTranslator(source=src_lang.lower(), target="en").translate(text)

def translate_from_english(text, dest_lang):
    if dest_lang.lower()=="english":
        return text
    return GoogleTranslator(source="en", target=dest_lang.lower()).translate(text)

def clean_ai_response(response_text):
    if not response_text:
        return response_text
    response_text = re.sub(r'<[^>]+>', '', response_text)
    response_text = re.sub(r'\s+', ' ', response_text).strip()
    response_text = response_text.replace('&nbsp;', ' ')
    response_text = response_text.replace('&lt;', '<')
    response_text = response_text.replace('&gt;', '>')
    response_text = response_text.replace('&amp;', '&')
    
    return response_text

def get_response_from_gemini(user_message, model):
    if model is None:
        return "I'm sorry, I can't connect right now. Please check the API configuration."

    prompt = f"""
     
    You are a multilingual healthcare assistant. The user will describe their symptoms in simple terms. 
    Your job is to:

    1. Summarize the most likely cause in simple, easy-to-understand language (no medical jargon).
    2. Suggest home remedies if applicable.
    3. Clearly state an Urgency Level as one of: Mild, Moderate, or Emergency.
    4. Keep the response under 100 words.
    5. Avoid listing many causes. Focus on the most probable one only.
    6. Add a disclaimer: "This is not a diagnosis. Consult a doctor if symptoms persist or worsen."

    Always respond in this structure:

    Possible Cause 
    Urgency Level 
    Suggestion 
    Note:This is not a diagnosis...

    The following is the user's message:
    User message: {user_message}
    
    if you see this , say "i see this"
    """
    try:
        response = model.generate_content(prompt)
        # Clean the response to remove any HTML or unwanted formatting
        cleaned_response = clean_ai_response(response.text)
        return cleaned_response
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return "Trouble connecting. Please try again later."

