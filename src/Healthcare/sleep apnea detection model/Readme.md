# 🩺 Sleep Apnea Detection from Snoring Audio

A deep learning-powered web application to detect **Sleep Apnea** from **snoring audio clips**.  
The model uses **Convolutional Neural Networks (CNNs)** trained on preprocessed audio signals, specifically **MFCC (Mel-Frequency Cepstral Coefficients)** features, to classify whether a snoring audio sample indicates **Normal** breathing or **Sleep Apnea**.

---
## Dataset 
https://www.kaggle.com/datasets/tareqkhanemu/snoring?utm_

## Features
-  Upload **1-second audio clips** in `.wav` or `.mp3` format.
-  Uses a trained **CNN model** for classification.
- Real-time prediction with confidence score.
-  Simple and user-friendly web app built with **Streamlit**.
-  Visualizes model performance and supports further improvements.

---

##  Tech Stack
- **Python** for data processing and machine learning.
- **TensorFlow/Keras** for building and training the CNN model.
- **Librosa** for audio processing and MFCC feature extraction.
- **Streamlit** for deploying the web app.

---

##  Project Structure
- sleep-apnea-detection
- app.py # Streamlit web app
- sleep_apnea_cnn_model.h5 # Trained CNN model
- README.md # Project documentation


---

##  How It Works
1. **Audio Preprocessing**
   - Audio is resampled to **16 kHz**.
   - Padded or trimmed to **1 second** duration.
   - Extract **MFCC features** (40 coefficients per frame).
   
2. **Model**
   - CNN trained on labeled snoring samples (`Normal` vs `Apnea`).
   - Output layer uses **Softmax** for classification.

3. **Prediction**
   - Upload audio through the Streamlit interface.
   - Model predicts class and confidence score.

---
## Author
GitHub Link: https://github.com/student-smritipandey
