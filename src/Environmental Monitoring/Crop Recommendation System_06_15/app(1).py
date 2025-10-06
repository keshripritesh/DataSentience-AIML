import streamlit as st
import joblib
import numpy as np

# Load model and encoder
model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

# Page config
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌿", layout="centered")

# Custom styling for dark mode compatibility
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        .main-box {
            background-color: #1e1e1e;
            padding: 2rem;
            border-radius: 12px;
            color: #f0f0f0;
        }
        h1, h2, h3, .stMarkdown {
            color: #f0f0f0 !important;
        }
        .stButton > button {
            background-color: #28a745;
            color: white;
            font-weight: bold;
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌿 Crop Recommendation System")

# Form layout
with st.container():
    with st.form("crop_form"):
        st.markdown('<div class="main-box">', unsafe_allow_html=True)
        st.subheader("📋 Enter Soil & Weather Conditions")

        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90)
            P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=42)
            K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=43)
            ph = st.slider("pH Level", 3.0, 10.0, 6.5)

        with col2:
            temperature = st.slider("Temperature (°C)", 0.0, 50.0, 20.8)
            humidity = st.slider("Humidity (%)", 10.0, 100.0, 82.0)
            rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 202.9)

        submitted = st.form_submit_button("🌾 Recommend Crop")
        st.markdown('</div>', unsafe_allow_html=True)

# Output
if submitted:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    predicted_crop = le.inverse_transform(prediction)[0]

    st.success(f"✅ **Recommended Crop:** `{predicted_crop}`")
    st.balloons()
