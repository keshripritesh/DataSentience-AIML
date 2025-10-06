import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("sleep_apnea_cnn_model.h5")

# Parameters
SR = 16000          # Sample rate
MAX_LEN = 1 * SR    # 1 second
N_MFCC = 40

# Function to pad or trim audio (same as training)
def pad_or_trim(audio, max_len=MAX_LEN):
    if len(audio) > max_len:
        return audio[:max_len]
    else:
        return np.pad(audio, (0, max_len - len(audio)))

# Function to extract MFCCs (same as training)
def extract_mfcc(audio, sr=SR, n_mfcc=N_MFCC):
    audio = audio.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # (time_steps, n_mfcc)

# Prediction function
def predict_audio(file_path, model):
    # Load and preprocess
    audio, _ = librosa.load(file_path, sr=SR)
    audio = pad_or_trim(audio, MAX_LEN)
    mfcc = extract_mfcc(audio, sr=SR, n_mfcc=N_MFCC)

    # Reshape for CNN
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # (1, time_steps, n_mfcc, 1)
    mfcc = mfcc.astype(np.float32)

    # Predict
    pred = model.predict(mfcc)
    predicted_class = np.argmax(pred, axis=1)[0]
    return predicted_class, pred

# Streamlit UI
st.title("🩺 Sleep Apnea Detection from Snoring Audio")
st.write("Upload a 1-second audio clip to check if it's Apnea or Normal.")

uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp_audio.wav")

    predicted_class, pred = predict_audio("temp_audio.wav", model)
    class_names = ["Normal", "Apnea"]  # Change if your order is different
    confidence = pred[0][predicted_class] * 100

    st.subheader(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")
