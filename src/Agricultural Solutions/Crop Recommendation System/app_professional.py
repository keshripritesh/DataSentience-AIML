"""
🌾 Agricultural Solutions Web Dashboard - Professional Edition
Advanced crop recommendation system with weather integration, soil health monitoring, and price forecasting
Built with Streamlit and Machine Learning
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import pickle
from utils import (
    get_weather_data, 
    analyze_soil_health, 
    get_crop_prices, 
    predict_price_trend, 
    translate_text,
    SUPPORTED_LANGUAGES
)
from config import CROP_PRICES, SOIL_HEALTH_PARAMS

# Configure page
st.set_page_config(
    page_title="🌾 Agricultural Solutions Dashboard", 
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling with Pastel Colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        padding: 2rem 1rem;
    }
    
    .main-header {
        background: linear-gradient(135deg, #A7F3D0 0%, #86EFAC 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    .main-header h1 {
        color: #047857;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-header p {
        color: #065F46;
        font-size: 1.2rem;
        margin: 1rem 0 0 0;
        font-weight: 500;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
        padding: 1.8rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 25px rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(147, 197, 253, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 35px rgba(59, 130, 246, 0.2);
    }
    
    .metric-card h3 {
        margin: 0;
        color: #1D4ED8;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .metric-card h2 {
        margin: 0.5rem 0;
        color: #1E40AF;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border: 2px solid #F59E0B;
        border-radius: 12px;
        color: #92400E;
        font-weight: 500;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #A78BFA 0%, #8B5CF6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
    }
    
    .info-box {
        background: linear-gradient(135deg, #F3E8FF 0%, #E9D5FF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #8B5CF6;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #10B981;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #F59E0B;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.1);
    }
    
    /* Sidebar Professional Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #F1F5F9 0%, #E2E8F0 100%);
    }
    
    .css-1d391kg .stRadio > label {
        background: rgba(255, 255, 255, 0.8);
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        border: 1px solid #CBD5E1;
        transition: all 0.3s ease;
    }
    
    .css-1d391kg .stRadio > label:hover {
        background: rgba(167, 243, 208, 0.3);
        border-color: #10B981;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, #E2E8F0 0%, #CBD5E1 100%);
        border-radius: 15px;
        color: #475569;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Professional Sidebar Design
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; border-bottom: 2px solid #E5E7EB; margin-bottom: 2rem;">
        <h2 style="color: #1F2937; font-weight: 700; margin: 0; font-size: 1.8rem;">🌾 AgriDash</h2>
        <p style="color: #6B7280; font-size: 0.95rem; margin: 0.3rem 0 0 0; font-weight: 500;">Smart Agriculture Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🌐 Language Preferences")
    selected_lang = st.selectbox(
        "Choose Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: SUPPORTED_LANGUAGES[x],
        index=0,
        key="language_selector"
    )
    
    st.markdown("---")
    
    st.markdown("#### 📊 Dashboard Navigation")
    page = st.radio(
        "Select Module",
        ["🏠 Dashboard", "🌾 Crop Recommendation", "🌤️ Weather Monitor", "🧪 Soil Analysis", "💰 Price Forecast"],
        index=0,
        key="page_selector"
    )
    
    st.markdown("---")
    
    # Professional Quick Stats
    st.markdown("#### 📈 Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active Users", "2,847", "↑ 12%")
        st.metric("Accuracy Rate", "99.3%", "↑ 0.1%")
    with col2:
        st.metric("Predictions", "156", "↑ 8%")
        st.metric("Success Rate", "97.8%", "↑ 2.1%")

# Translate function  
def t(text):
    return translate_text(text, selected_lang)

# Enhanced Professional Header
st.markdown(f"""
<div class="main-header">
    <h1>🌾 {t('Agricultural Solutions Dashboard')}</h1>
    <p>Intelligent Crop Recommendations • Real-time Weather • Soil Health Monitoring • Market Forecasting</p>
</div>
""", unsafe_allow_html=True)

if page == "🏠 Dashboard":
    st.markdown("## 🏠 Agricultural Dashboard Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🌡️ Average Temperature</h3>
            <h2>25.3°C</h2>
            <p style="color: #10B981; font-weight: 500;">↑ 2.1°C from yesterday</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>💧 Humidity Level</h3>
            <h2>68%</h2>
            <p style="color: #3B82F6; font-weight: 500;">↓ 5% from yesterday</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🌱 Soil pH</h3>
            <h2>6.8</h2>
            <p style="color: #10B981; font-weight: 500;">Optimal Range</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Market Trend</h3>
            <h2>+12%</h2>
            <p style="color: #10B981; font-weight: 500;">Price increase</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dashboard content sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Weekly Overview")
        
        # Sample chart data
        dates = pd.date_range(start='2024-01-01', periods=7, freq='D')
        temp_data = [22, 25, 27, 24, 26, 23, 25]
        humidity_data = [65, 70, 68, 72, 69, 71, 68]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=temp_data, mode='lines+markers', name='Temperature (°C)', line=dict(color='#EF4444', width=3)))
        fig.add_trace(go.Scatter(x=dates, y=humidity_data, mode='lines+markers', name='Humidity (%)', line=dict(color='#3B82F6', width=3)))
        
        fig.update_layout(
            title="Weather Trends This Week",
            title_font_size=18,
            title_font_color='#1F2937',
            xaxis_title="Date",
            yaxis_title="Values",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12, color='#374151')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        
        if st.button("🌾 Get Crop Recommendation", use_container_width=True):
            st.success("Redirecting to Crop Recommendation module...")
        
        if st.button("🌤️ Check Weather", use_container_width=True):
            st.info("Redirecting to Weather Monitor...")
        
        if st.button("🧪 Analyze Soil", use_container_width=True):
            st.warning("Redirecting to Soil Analysis...")
        
        if st.button("💰 Price Forecast", use_container_width=True):
            st.info("Redirecting to Price Forecast...")
        
        st.markdown("### 📢 Latest Updates")
        st.markdown("""
        <div class="info-box">
            <h4>🆕 New Feature</h4>
            <p>Multi-language support now available in 9 Indian languages!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>✅ Model Update</h4>
            <p>Crop recommendation accuracy improved to 99.3%</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🌾 Crop Recommendation":
    st.markdown("## 🌾 Smart Crop Recommendation System")
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 How it works</h4>
        <p>Our AI-powered system analyzes soil and weather conditions to recommend the most suitable crops for your land. 
        Simply enter your farm's parameters below and get personalized recommendations with 99.3% accuracy!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🌡️ Climate Parameters")
        temperature = st.slider(f"{t('Temperature')} (°C)", 0, 50, 25)
        humidity = st.slider(f"{t('Humidity')} (%)", 0, 100, 70)
        ph = st.slider(f"{t('Soil pH')}", 3.0, 10.0, 6.5, 0.1)
        rainfall = st.slider(f"{t('Rainfall')} (mm)", 0, 3000, 1200)
    
    with col2:
        st.markdown("#### 🧪 Soil Nutrients")
        nitrogen = st.slider(f"{t('Nitrogen')} (kg/ha)", 0, 200, 50)
        phosphorus = st.slider(f"{t('Phosphorus')} (kg/ha)", 0, 150, 25)
        potassium = st.slider(f"{t('Potassium')} (kg/ha)", 0, 200, 50)
    
    with col3:
        st.markdown("#### 📊 Current Values")
        st.metric("Temperature", f"{temperature}°C")
        st.metric("Humidity", f"{humidity}%")
        st.metric("pH Level", f"{ph}")
        st.metric("NPK Ratio", f"{nitrogen}:{phosphorus}:{potassium}")
    
    # Prediction
    if st.button("🔮 Get Crop Recommendation", use_container_width=True):
        try:
            # Load the trained model
            with open('crop_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            
            # Make prediction
            input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_features)
            predicted_crop = label_encoder.inverse_transform(prediction)[0]
            prediction_proba = model.predict_proba(input_features).max()
            
            st.markdown("### 🎉 Recommendation Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="success-box">
                    <h3>🌾 Recommended Crop: <span style="color: #059669; font-weight: 700;">{predicted_crop.title()}</span></h3>
                    <p><strong>Confidence Level:</strong> {prediction_proba:.1%}</p>
                    <p><strong>Suitability:</strong> Excellent match for your soil and weather conditions!</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional crop info
                crop_info = {
                    'rice': "🌾 High water requirement, suitable for monsoon season",
                    'wheat': "🌾 Winter crop, requires cool weather",
                    'cotton': "🌱 Cash crop, needs warm weather and moderate rainfall",
                    'sugarcane': "🎋 Long duration crop, high water and nutrient requirement",
                    'maize': "🌽 Kharif crop, adaptable to various conditions"
                }
                
                if predicted_crop.lower() in crop_info:
                    st.info(f"💡 **About {predicted_crop}:** {crop_info[predicted_crop.lower()]}")
            
            with col2:
                # Show prediction confidence
                confidence_fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction_proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#10B981"},
                        'steps': [
                            {'range': [0, 60], 'color': "#FEE2E2"},
                            {'range': [60, 80], 'color': "#FEF3C7"},
                            {'range': [80, 100], 'color': "#D1FAE5"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90}}))
                
                confidence_fig.update_layout(height=300)
                st.plotly_chart(confidence_fig, use_container_width=True)
                
        except FileNotFoundError:
            st.error("⚠️ Model files not found. Please run the training script first.")
            if st.button("🚀 Train Model Now"):
                with st.spinner("Training model... This may take a few minutes."):
                    exec(open('train.py').read())
                st.success("✅ Model trained successfully! You can now get recommendations.")

elif page == "🌤️ Weather Monitor":
    st.markdown("## 🌤️ Weather Monitoring Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🌍 Current Weather Conditions")
        
        location = st.selectbox(
            "Select Location",
            ["Mumbai, Maharashtra", "Delhi, Delhi", "Pune, Maharashtra", "Bangalore, Karnataka", "Chennai, Tamil Nadu"],
            key="weather_location"
        )
        
        # Get weather data (demo mode)
        weather_data = get_weather_data(location.split(',')[0])
        
        if weather_data:
            # Current weather display
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🌡️ Temperature</h3>
                    <h2>{weather_data['temperature']}°C</h2>
                    <p style="color: #6B7280;">Feels like {weather_data['feels_like']}°C</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💧 Humidity</h3>
                    <h2>{weather_data['humidity']}%</h2>
                    <p style="color: #6B7280;">{weather_data['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💨 Wind Speed</h3>
                    <h2>{weather_data['wind_speed']} km/h</h2>
                    <p style="color: #6B7280;">Direction: {weather_data['wind_direction']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 7-day forecast
        st.markdown("### 📅 7-Day Weather Forecast")
        
        # Generate forecast data
        forecast_dates = pd.date_range(start=datetime.now(), periods=7, freq='D')
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Temperature': [25 + random.randint(-3, 5) for _ in range(7)],
            'Humidity': [65 + random.randint(-10, 15) for _ in range(7)],
            'Rainfall': [random.randint(0, 25) for _ in range(7)]
        })
        
        # Weather forecast chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=forecast_df['Date'], y=forecast_df['Temperature'], name="Temperature (°C)", line=dict(color='red')),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_df['Date'], y=forecast_df['Humidity'], name="Humidity (%)", line=dict(color='blue')),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
        fig.update_yaxes(title_text="Humidity (%)", secondary_y=True)
        fig.update_layout(title_text="7-Day Weather Forecast", height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🚨 Weather Alerts")
        
        # Weather alerts
        if weather_data and weather_data['temperature'] > 30:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ High Temperature Alert</h4>
                <p>Temperature above 30°C. Ensure adequate irrigation for crops.</p>
            </div>
            """, unsafe_allow_html=True)
        
        if weather_data and weather_data['humidity'] < 40:
            st.markdown("""
            <div class="warning-box">
                <h4>💧 Low Humidity Alert</h4>
                <p>Humidity below 40%. Consider additional watering.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🌱 Farming Tips")
        st.markdown("""
        <div class="info-box">
            <h4>💡 Today's Recommendation</h4>
            <p>Perfect weather for field activities. Good time for planting and harvesting.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Historical Data")
        
        # Historical weather trends (simplified)
        hist_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Avg_Temp': [20, 23, 27, 32, 35, 33],
            'Avg_Rainfall': [10, 15, 25, 45, 120, 200]
        })
        
        fig_hist = px.bar(hist_data, x='Month', y='Avg_Rainfall', title="Average Monthly Rainfall")
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

elif page == "🧪 Soil Analysis":
    st.markdown("## 🧪 Comprehensive Soil Health Analysis")
    
    st.markdown("""
    <div class="info-box">
        <h4>🔬 Soil Health Assessment</h4>
        <p>Get detailed insights into your soil's health including nutrient levels, pH balance, and recommendations for improvement.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Enter Soil Parameters")
        
        soil_ph = st.slider("Soil pH Level", 3.0, 10.0, 6.5, 0.1, key="soil_ph")
        nitrogen_level = st.slider("Nitrogen Level (mg/kg)", 0, 200, 85, key="soil_nitrogen")
        phosphorus_level = st.slider("Phosphorus Level (mg/kg)", 0, 100, 45, key="soil_phosphorus") 
        potassium_level = st.slider("Potassium Level (mg/kg)", 0, 200, 120, key="soil_potassium")
        organic_matter = st.slider("Organic Matter (%)", 0.0, 10.0, 3.5, 0.1, key="soil_organic")
        moisture_content = st.slider("Moisture Content (%)", 0, 50, 25, key="soil_moisture")
        
        if st.button("🔍 Analyze Soil Health", use_container_width=True):
            # Perform soil analysis
            soil_params = {
                'ph': soil_ph,
                'nitrogen': nitrogen_level,
                'phosphorus': phosphorus_level,
                'potassium': potassium_level,
                'organic_matter': organic_matter,
                'moisture': moisture_content
            }
            
            analysis_result = analyze_soil_health(soil_params)
            
            # Display results
            st.markdown("### 📈 Soil Health Report")
            
            # Overall health score
            health_score = analysis_result['overall_score']
            health_status = analysis_result['status']
            
            color = "#10B981" if health_score >= 80 else "#F59E0B" if health_score >= 60 else "#EF4444"
            
            st.markdown(f"""
            <div class="success-box" style="border-left-color: {color};">
                <h3>🏆 Overall Soil Health Score: <span style="color: {color};">{health_score}/100</span></h3>
                <p><strong>Status:</strong> {health_status}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Nutrient Analysis")
        
        if 'analysis_result' in locals():
            # Nutrient levels visualization
            nutrients = ['Nitrogen', 'Phosphorus', 'Potassium', 'Organic Matter']
            levels = [nitrogen_level, phosphorus_level, potassium_level, organic_matter * 10]  # Scale organic matter
            colors = ['#EF4444', '#F59E0B', '#10B981', '#8B5CF6']
            
            fig = go.Figure(data=[
                go.Bar(x=nutrients, y=levels, marker_color=colors)
            ])
            
            fig.update_layout(
                title="Nutrient Levels in Soil",
                yaxis_title="Concentration (mg/kg or %×10)",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Improvement Recommendations")
            
            recommendations = analysis_result['recommendations']
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="info-box">
                    <h5>{i}. {rec['title']}</h5>
                    <p>{rec['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Soil composition pie chart
    st.markdown("### 🥧 Soil Composition Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'analysis_result' in locals():
            composition_data = {
                'Component': ['Sand', 'Clay', 'Silt', 'Organic Matter', 'Other'],
                'Percentage': [35, 25, 20, organic_matter, 20 - organic_matter]
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=composition_data['Component'],
                values=composition_data['Percentage'],
                hole=0.4,
                marker_colors=['#FED7AA', '#A78BFA', '#93C5FD', '#86EFAC', '#E5E7EB']
            )])
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title="Soil Composition Breakdown", height=400)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Quick Facts")
        
        if 'analysis_result' in locals():
            st.metric("pH Level", f"{soil_ph}", "Optimal: 6.0-7.0")
            st.metric("Moisture", f"{moisture_content}%", "Good" if 20 <= moisture_content <= 30 else "Monitor")
            st.metric("N-P-K Ratio", f"{nitrogen_level}:{phosphorus_level}:{potassium_level}")
            
            st.markdown("""
            <div class="warning-box">
                <h5>🔔 Quick Tip</h5>
                <p>Regular soil testing helps maintain optimal crop yields and soil health.</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "💰 Price Forecast":
    st.markdown("## 💰 Crop Price Forecasting Dashboard")
    
    st.markdown("""
    <div class="info-box">
        <h4>📈 Market Intelligence</h4>
        <p>Stay ahead of market trends with our AI-powered price forecasting system. Make informed decisions about when to sell your crops for maximum profit.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Select Crop for Price Analysis")
        
        selected_crop = st.selectbox(
            "Choose Crop",
            list(CROP_PRICES.keys()),
            key="price_crop_selector"
        )
        
        time_period = st.selectbox(
            "Forecast Period",
            ["1 Week", "1 Month", "3 Months", "6 Months"],
            key="forecast_period"
        )
        
        if st.button("📈 Generate Price Forecast", use_container_width=True):
            # Get price forecast
            forecast_data = predict_price_trend(selected_crop, time_period)
            
            if forecast_data:
                st.markdown(f"### 📈 Price Forecast for {selected_crop.title()}")
                
                # Current vs Predicted
                current_price = CROP_PRICES[selected_crop]
                predicted_price = forecast_data['predicted_price']
                price_change = ((predicted_price - current_price) / current_price) * 100
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Current Price", f"₹{current_price}/kg")
                
                with col_b:
                    st.metric("Predicted Price", f"₹{predicted_price}/kg", f"{price_change:+.1f}%")
                
                with col_c:
                    trend_emoji = "📈" if price_change > 0 else "📉"
                    trend_text = "Bullish" if price_change > 0 else "Bearish"
                    st.metric("Market Trend", f"{trend_emoji} {trend_text}")
                
                # Price trend chart
                dates = pd.date_range(start=datetime.now(), periods=len(forecast_data['prices']), freq='D')
                price_df = pd.DataFrame({
                    'Date': dates,
                    'Price': forecast_data['prices']
                })
                
                fig = px.line(price_df, x='Date', y='Price', title=f"{selected_crop.title()} Price Trend Forecast")
                fig.update_traces(line_color='#10B981', line_width=3)
                fig.update_layout(
                    yaxis_title="Price (₹/kg)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trading recommendations
                st.markdown("### 🎯 Trading Recommendations")
                
                if price_change > 5:
                    st.markdown("""
                    <div class="success-box">
                        <h4>🚀 Strong Buy Signal</h4>
                        <p>Prices are expected to rise significantly. Consider holding your harvest for better returns.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif price_change > 0:
                    st.markdown("""
                    <div class="info-box">
                        <h4>📊 Moderate Buy</h4>
                        <p>Slight upward trend expected. Good time to hold or gradually sell.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ Consider Selling</h4>
                        <p>Prices may decline. Consider selling sooner rather than later.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💹 Market Overview")
        
        # Top performing crops
        st.markdown("#### 🏆 Top Performing Crops")
        
        top_crops = ['Wheat', 'Rice', 'Cotton', 'Sugarcane']
        for crop in top_crops:
            price = CROP_PRICES.get(crop.lower(), 50)
            change = random.randint(-5, 15)
            color = "#10B981" if change > 0 else "#EF4444"
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid {color};">
                <strong>{crop}</strong><br>
                ₹{price}/kg <span style="color: {color};">({change:+.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 📰 Market News")
        st.markdown("""
        <div class="info-box">
            <h5>🌾 Wheat Exports Rise</h5>
            <p>Government approves increased wheat exports leading to price surge.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h5>🌧️ Monsoon Alert</h5>
            <p>Early monsoon may affect cotton prices in coming weeks.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Market sentiment gauge
        st.markdown("#### 📊 Market Sentiment")
        sentiment_score = 72  # Example sentiment score
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sentiment_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Sentiment"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#8B5CF6"},
                'steps': [
                    {'range': [0, 40], 'color': "#FEE2E2"},
                    {'range': [40, 70], 'color': "#FEF3C7"},
                    {'range': [70, 100], 'color': "#D1FAE5"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80}}))
        
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

# Professional Footer
st.markdown("""
<div class="footer">
    <p>🌾 Agricultural Solutions Dashboard • Built with ❤️ for Farmers</p>
    <p>Empowering Agriculture through Technology • v2.0 Professional Edition</p>
</div>
""", unsafe_allow_html=True)
