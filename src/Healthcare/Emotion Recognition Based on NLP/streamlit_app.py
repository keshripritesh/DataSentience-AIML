import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import time
from datetime import datetime
import json
import os
import re

# Page Configuration
st.set_page_config(
    page_title="🧠 Advanced Emotion Recognition Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for healthcare-grade styling with dynamic colors
st.markdown("""
    <style>
        @keyframes emotionPulse {
            0% { opacity: 0.6; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.02); }
            100% { opacity: 0.6; transform: scale(1); }
        }
        
        @keyframes colorShift {
            0% { filter: hue-rotate(0deg); }
            25% { filter: hue-rotate(10deg); }
            50% { filter: hue-rotate(0deg); }
            75% { filter: hue-rotate(-10deg); }
            100% { filter: hue-rotate(0deg); }
        }
        
        .main-header {
            background: linear-gradient(135deg, #e8f4fd 0%, #f0f8ff 50%, #fff5f5 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            color: #2c3e50;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.2);
            animation: colorShift 5s ease-in-out infinite;
        }
        
        .emotion-card {
            background: linear-gradient(145deg, #ffffff 0%, #fafafa 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            margin: 1rem 0;
            border-left: 4px solid #4CAF50;
            border-top: 1px solid rgba(76, 175, 80, 0.3);
            transition: all 0.3s ease;
        }
        
        .emotion-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        }
        
        .dynamic-emotion-card {
            background: linear-gradient(145deg, #ffffff 0%, #fafafa 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            margin: 1rem 0;
            border-left: 4px solid var(--emotion-color, #4CAF50);
            border-top: 1px solid var(--emotion-color-transparent, rgba(76, 175, 80, 0.3));
            animation: emotionPulse 3s ease-in-out infinite;
            transition: all 0.3s ease;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            color: #2c3e50;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.8rem 0;
            box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.6);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 18px rgba(0, 0, 0, 0.12);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #81C784 0%, #A5D6A7 50%, #C8E6C9 100%) !important;
            color: #2E7D32 !important;
            border: 2px solid #66BB6A !important;
            border-radius: 25px !important;
            padding: 0.75rem 2rem !important;
            font-weight: bold !important;
            font-size: 1.1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(129, 199, 132, 0.3) !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #66BB6A 0%, #81C784 50%, #A5D6A7 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(129, 199, 132, 0.4) !important;
        }
        
        .success-message {
            background: linear-gradient(135deg, #d4edda 0%, #e8f5e9 100%);
            color: #155724;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 4px solid #28a745;
            box-shadow: 0 3px 10px rgba(40, 167, 69, 0.1);
            animation: emotionPulse 4s ease-in-out infinite;
        }
        
        .warning-message {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 4px solid #ffc107;
            box-shadow: 0 3px 10px rgba(255, 193, 7, 0.1);
        }
        
        .info-message {
            background: linear-gradient(135deg, #d1ecf1 0%, #e2f3f5 100%);
            color: #0c5460;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 4px solid #17a2b8;
            box-shadow: 0 3px 10px rgba(23, 162, 184, 0.1);
        }
        
        .confidence-bar {
            height: 20px;
            border-radius: 10px;
            transition: all 0.5s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

class SmartEmotionPredictor:
    """Smart emotion predictor with negation handling"""
    
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        
    def detect_negation(self, text):
        """Detect if text contains negation patterns"""
        negation_patterns = [
            r'\b(not|don\'t|doesn\'t|didn\'t|won\'t|can\'t|isn\'t|aren\'t|wasn\'t|weren\'t)\b',
            r'\b(never|no|nobody|nothing|nowhere|neither|nor)\b',
            r'\b(hardly|barely|scarcely)\b'
        ]
        
        text_lower = text.lower()
        for pattern in negation_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        text = text.lower()
        # Keep negations as important markers
        text = re.sub(r"don't", "do_not", text)
        text = re.sub(r"doesn't", "does_not", text)  
        text = re.sub(r"didn't", "did_not", text)
        text = re.sub(r"won't", "will_not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"isn't", "is_not", text)
        text = re.sub(r"aren't", "are_not", text)
        return text
    
    def predict_emotion(self, text):
        """Predict emotion with smart negation handling"""
        # Check for negation
        has_negation = self.detect_negation(text)
        
        # Preprocess and vectorize
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        
        # Get prediction and probabilities
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        # Get all emotion probabilities
        emotion_probs = {}
        for i, emotion in enumerate(self.model.classes_):
            emotion_probs[emotion] = probabilities[i] * 100
        
        # Apply negation logic for specific cases
        if has_negation:
            # Check if the predicted emotion makes sense with negation
            negative_indicators = ['good', 'great', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'enjoy']
            positive_indicators = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'scared']
            
            text_lower = text.lower()
            
            # If negation + positive word -> likely negative emotion
            if any(word in text_lower for word in negative_indicators):
                if prediction in ['joy', 'love', 'surprise']:
                    # Override with more appropriate negative emotion
                    if 'feel' in text_lower or 'good' in text_lower:
                        prediction = 'sadness'
                    elif 'like' in text_lower or 'love' in text_lower:
                        prediction = 'anger'
                    
                    confidence = 0.85  # High confidence in negation override
                    
                    # Adjust probabilities
                    emotion_probs = {emotion: 10.0 for emotion in emotion_probs}
                    emotion_probs[prediction] = 85.0
        
        return prediction, confidence * 100, emotion_probs

# Load the trained model and vectorizer
@st.cache_resource
def load_models():
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to load smart model first, fallback to regular model
        try:
            model_path = os.path.join(script_dir, 'nlp_smart.pkl')
            transform_path = os.path.join(script_dir, 'transform_smart.pkl')
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(transform_path, 'rb') as f:
                vectorizer = pickle.load(f)
            return model, vectorizer, True  # Smart model loaded
        except:
            # Fallback to regular model
            model_path = os.path.join(script_dir, 'nlp.pkl')
            transform_path = os.path.join(script_dir, 'transform.pkl')
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(transform_path, 'rb') as f:
                vectorizer = pickle.load(f)
            return model, vectorizer, False  # Regular model loaded
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, False

# Emotion colors and emojis
# Emotion configuration with dynamic colors and minimal emojis
EMOTION_CONFIG = {
    'joy': {'color': '#4CAF50', 'emoji': '😊', 'description': 'Happiness, positivity, contentment'},
    'sadness': {'color': '#2196F3', 'emoji': '😢', 'description': 'Melancholy, sorrow, disappointment'},
    'anger': {'color': '#F44336', 'emoji': '😠', 'description': 'Frustration, irritation, rage'},
    'fear': {'color': '#9C27B0', 'emoji': '😨', 'description': 'Anxiety, worry, apprehension'},
    'love': {'color': '#E91E63', 'emoji': '❤️', 'description': 'Affection, care, romantic feelings'},
    'surprise': {'color': '#FF9800', 'emoji': '😲', 'description': 'Astonishment, amazement, shock'}
}

def get_dynamic_color(emotion, confidence):
    """Get dynamic color based on emotion and confidence"""
    base_color = EMOTION_CONFIG[emotion]['color']
    
    # Convert hex to RGB and apply confidence-based opacity
    hex_color = base_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Adjust intensity based on confidence
    intensity = 0.4 + (confidence * 0.6)  # Range from 40% to 100%
    
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {intensity})"

def predict_emotion(text, model, vectorizer, is_smart=False):
    """Predict emotion with confidence scores using smart or regular model"""
    if not text.strip():
        return None, None
    
    try:
        if is_smart:
            # Use smart predictor
            predictor = SmartEmotionPredictor(model, vectorizer)
            prediction, confidence, all_probs = predictor.predict_emotion(text)
            
            # Convert to format expected by the app
            confidence_scores = {emotion: prob/100 for emotion, prob in all_probs.items()}
        else:
            # Use regular prediction
            text_vector = vectorizer.transform([text])
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
            
            # Get confidence scores for all emotions
            emotions = model.classes_
            confidence_scores = {emotion: prob for emotion, prob in zip(emotions, probabilities)}
        
        return prediction, confidence_scores
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def create_emotion_gauge(emotion, confidence):
    """Create emotion confidence gauge"""
    color = EMOTION_CONFIG.get(emotion, {}).get('color', '#666666')
    emoji = EMOTION_CONFIG.get(emotion, {}).get('emoji', '🤔')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{emoji} {emotion.title()} Confidence"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=300, font={'size': 12})
    return fig

def create_confidence_chart(confidence_scores):
    """Create horizontal bar chart for all emotion confidences"""
    emotions = list(confidence_scores.keys())
    confidences = [confidence_scores[emotion] * 100 for emotion in emotions]
    colors = [EMOTION_CONFIG.get(emotion, {}).get('color', '#666666') for emotion in emotions]
    
    fig = go.Figure(go.Bar(
        x=confidences,
        y=emotions,
        orientation='h',
        marker_color=colors,
        text=[f"{conf:.1f}%" for conf in confidences],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="🎭 Emotion Confidence Scores",
        xaxis_title="Confidence (%)",
        yaxis_title="Emotions",
        height=400,
        showlegend=False
    )
    return fig

def get_mental_health_insights(emotion, confidence):
    """Get mental health insights based on detected emotion"""
    insights = {
        'joy': {
            'tips': [
                "💫 Great! Positive emotions boost mental health",
                "🌟 Share your joy with others to amplify it",
                "🎯 Use this positive energy for goal achievement",
                "🧘 Practice gratitude to maintain this feeling"
            ],
            'recommendations': [
                "Continue activities that bring you joy",
                "Consider journaling about positive experiences",
                "Engage in social activities to share happiness"
            ]
        },
        'sadness': {
            'tips': [
                "🤗 It's okay to feel sad - emotions are natural",
                "💙 Reach out to friends or family for support",
                "🏃 Physical activity can help improve mood",
                "🎵 Listen to uplifting music or engage in hobbies"
            ],
            'recommendations': [
                "Consider talking to a mental health professional",
                "Practice self-care and mindfulness",
                "Maintain social connections"
            ]
        },
        'anger': {
            'tips': [
                "🔥 Take deep breaths to calm down",
                "🚶 Try physical exercise to release tension",
                "✍️ Write down your feelings to process them",
                "🧘 Practice mindfulness or meditation"
            ],
            'recommendations': [
                "Learn anger management techniques",
                "Identify triggers and develop coping strategies",
                "Consider counseling if anger is frequent"
            ]
        },
        'fear': {
            'tips': [
                "🛡️ Fear is natural - acknowledge it without judgment",
                "💪 Break down fears into manageable steps",
                "🤝 Seek support from trusted people",
                "🧘 Practice relaxation techniques"
            ],
            'recommendations': [
                "Gradually expose yourself to feared situations",
                "Learn anxiety management techniques",
                "Consider professional help for persistent fears"
            ]
        },
        'love': {
            'tips': [
                "❤️ Love is powerful - cherish these feelings",
                "💝 Express your feelings appropriately",
                "🌱 Nurture relationships that matter to you",
                "🎨 Channel love into creative expressions"
            ],
            'recommendations': [
                "Communicate openly in relationships",
                "Practice self-love and self-care",
                "Build healthy relationship boundaries"
            ]
        },
        'surprise': {
            'tips': [
                "⚡ Surprise keeps life interesting!",
                "🎯 Use unexpected moments for growth",
                "🤔 Reflect on what the surprise teaches you",
                "🌟 Embrace change and new experiences"
            ],
            'recommendations': [
                "Stay open to new experiences",
                "Practice adaptability and flexibility",
                "Use surprises as learning opportunities"
            ]
        }
    }
    
    return insights.get(emotion, {
        'tips': ["🧠 Every emotion provides valuable information"],
        'recommendations': ["Consider professional guidance for persistent concerns"]
    })

# Header
st.markdown("""
    <div class="main-header">
        <h1>🧠 Advanced Emotion Recognition Platform</h1>
        <p style="font-size: 1.2rem; margin: 0.5rem 0;">AI-Powered Mental Health & Emotion Analysis</p>
        <p style="font-size: 1rem; opacity: 0.8;">✨ Real-time emotion detection with professional insights ✨</p>
    </div>
""", unsafe_allow_html=True)

# Load models
classifier, cv, is_smart = load_models()

if classifier is None or cv is None:
    st.error("❌ Models not loaded. Please ensure model files exist.")
    st.stop()

# Display model type
if is_smart:
    st.success("🧠 Smart Emotion Model Loaded - Advanced negation handling enabled!")
else:
    st.warning("⚠️ Basic model loaded - Consider training the smart model for better accuracy.")

# Sidebar
with st.sidebar:
    st.header("🎛️ Analysis Controls")
    
    analysis_mode = st.selectbox(
        "📊 Analysis Mode",
        ["Single Text", "Batch Analysis", "Real-time Stream"]
    )
    
    st.header("📈 Model Information")
    st.info("""
    **🤖 Current Model:**
    - Algorithm: Multinomial Naive Bayes
    - Features: CountVectorizer (16,798 features)
    - Accuracy: 77.53%
    - Emotions: 6 classes
    """)
    
    st.header("🎭 Emotion Guide")
    for emotion, config in EMOTION_CONFIG.items():
        st.write(f"{config['emoji']} **{emotion.title()}**: {config['description']}")
    
    st.header("🔗 Quick Actions")
    if st.button("🧹 Clear History"):
        if 'emotion_history' in st.session_state:
            st.session_state.emotion_history = []
        st.success("History cleared!")

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

# Main content based on analysis mode
if analysis_mode == "Single Text":
    st.header("📝 Single Text Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
        st.subheader("✍️ Enter Your Text")
        
        text_input = st.text_area(
            "Text to analyze:",
            height=150,
            placeholder="Type or paste your text here... (e.g., 'I'm feeling really happy today!')",
            help="Enter any text to analyze its emotional content"
        )
        
        analyze_button = st.button(
            "🔍 Analyze Emotion",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if analyze_button and text_input:
            with st.spinner("🧠 Analyzing emotions..."):
                emotion, confidence_scores = predict_emotion(text_input, classifier, cv, is_smart)
                
                if emotion and confidence_scores:
                    # Store in history
                    st.session_state.emotion_history.append({
                        'timestamp': datetime.now(),
                        'text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                        'emotion': emotion,
                        'confidence': confidence_scores[emotion]
                    })
                    
                    # Main result
                    st.markdown(f"""
                        <div class="success-message">
                            <h3>🎭 Detected Emotion: {EMOTION_CONFIG[emotion]['emoji']} {emotion.title()}</h3>
                            <p><strong>Confidence:</strong> {confidence_scores[emotion]*100:.1f}%</p>
                            <p><strong>Description:</strong> {EMOTION_CONFIG[emotion]['description']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence chart
                    st.plotly_chart(create_confidence_chart(confidence_scores), use_container_width=True)
                    
                    # Mental health insights
                    insights = get_mental_health_insights(emotion, confidence_scores[emotion])
                    
                    st.subheader("💡 Mental Health Insights")
                    
                    col_tips, col_rec = st.columns(2)
                    
                    with col_tips:
                        st.markdown("**🧠 Emotional Wellness Tips:**")
                        for tip in insights['tips']:
                            st.write(f"• {tip}")
                    
                    with col_rec:
                        st.markdown("**🎯 Recommendations:**")
                        for rec in insights['recommendations']:
                            st.write(f"• {rec}")
    
    with col2:
        st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
        st.subheader("📊 Live Preview")
        
        if text_input:
            # Real-time analysis preview
            emotion, confidence_scores = predict_emotion(text_input, classifier, cv, is_smart)
            
            if emotion and confidence_scores:
                # Emotion gauge
                fig_gauge = create_emotion_gauge(emotion, confidence_scores[emotion])
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Quick stats
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 Primary Emotion</h4>
                        <h2>{EMOTION_CONFIG[emotion]['emoji']} {emotion.title()}</h2>
                        <p>{confidence_scores[emotion]*100:.1f}% confidence</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("👆 Enter text to see live analysis")
        else:
            st.info("👆 Enter text to see live analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif analysis_mode == "Batch Analysis":
    st.header("📄 Batch Text Analysis")
    
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    
    # File upload option
    uploaded_file = st.file_uploader(
        "📁 Upload a text file",
        type=['txt', 'csv'],
        help="Upload a text file or CSV with text column"
    )
    
    # Manual text entry
    st.subheader("✍️ Or enter multiple texts manually")
    batch_text = st.text_area(
        "Enter texts (one per line):",
        height=200,
        placeholder="Line 1: I'm so happy today!\nLine 2: This is really frustrating...\nLine 3: I love spending time with family"
    )
    
    if st.button("🔍 Analyze Batch", use_container_width=True):
        texts_to_analyze = []
        
        # Process uploaded file
        if uploaded_file:
            try:
                if uploaded_file.type == "text/plain":
                    content = uploaded_file.read().decode('utf-8')
                    texts_to_analyze.extend([line.strip() for line in content.split('\n') if line.strip()])
                elif uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    if 'text' in df.columns:
                        texts_to_analyze.extend(df['text'].astype(str).tolist())
                    else:
                        st.error("CSV must have a 'text' column")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Process manual text
        if batch_text:
            texts_to_analyze.extend([line.strip() for line in batch_text.split('\n') if line.strip()])
        
        if texts_to_analyze:
            with st.spinner(f"🧠 Analyzing {len(texts_to_analyze)} texts..."):
                results = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(texts_to_analyze):
                    emotion, confidence_scores = predict_emotion(text, classifier, cv, is_smart)
                    if emotion and confidence_scores:
                        results.append({
                            'Text': text[:100] + "..." if len(text) > 100 else text,
                            'Emotion': emotion,
                            'Confidence': f"{confidence_scores[emotion]*100:.1f}%",
                            'Emoji': EMOTION_CONFIG[emotion]['emoji']
                        })
                    progress_bar.progress((i + 1) / len(texts_to_analyze))
                
                if results:
                    # Display results table
                    df_results = pd.DataFrame(results)
                    st.subheader(f"📊 Analysis Results ({len(results)} texts)")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Emotion distribution
                    emotion_counts = df_results['Emotion'].value_counts()
                    
                    fig_pie = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="🎭 Emotion Distribution",
                        color_discrete_map={emotion: config['color'] for emotion, config in EMOTION_CONFIG.items()}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        "emotion_analysis_results.csv",
                        "text/csv"
                    )
    
    st.markdown('</div>', unsafe_allow_html=True)

elif analysis_mode == "Real-time Stream":
    st.header("🌊 Real-time Emotion Stream")
    
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-message">
            <h4>🔄 Real-time Analysis Mode</h4>
            <p>Type in the text area below and watch emotions being detected in real-time!</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Real-time text input
    stream_text = st.text_area(
        "🎤 Start typing for real-time analysis:",
        height=100,
        key="stream_input"
    )
    
    if stream_text:
        emotion, confidence_scores = predict_emotion(stream_text, classifier, cv, is_smart)
        
        if emotion and confidence_scores:
            col1, col2 = st.columns(2)
            
            with col1:
                # Real-time gauge
                fig_gauge = create_emotion_gauge(emotion, confidence_scores[emotion])
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Real-time confidence chart
                fig_conf = create_confidence_chart(confidence_scores)
                st.plotly_chart(fig_conf, use_container_width=True)
            
            # Current emotion display with dynamic colors
            dynamic_color = get_dynamic_color(emotion, confidence_scores[emotion])
            st.markdown(f"""
                <div class="success-message" style="background: {dynamic_color}; border-left-color: {EMOTION_CONFIG[emotion]['color']};">
                    <h3>Current Emotion: <strong>{emotion.title()}</strong> {EMOTION_CONFIG[emotion]['emoji']}</h3>
                    <p><strong>Confidence:</strong> {confidence_scores[emotion]*100:.1f}%</p>
                    <p style="font-size: 0.9em; opacity: 0.8;">{EMOTION_CONFIG[emotion]['description']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Emotion History
if st.session_state.emotion_history:
    st.header("📈 Emotion History")
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(st.session_state.emotion_history)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Recent emotions
        st.subheader("🕒 Recent Analysis")
        for entry in st.session_state.emotion_history[-5:]:
            st.markdown(f"""
                <div class="emotion-card">
                    <p><strong>{entry['timestamp'].strftime('%H:%M:%S')}</strong></p>
                    <p>{EMOTION_CONFIG[entry['emotion']]['emoji']} <strong>{entry['emotion'].title()}</strong> ({entry['confidence']*100:.1f}%)</p>
                    <p style="font-size: 0.9em; color: #666;">{entry['text']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Emotion timeline
        if len(history_df) > 1:
            fig_timeline = px.line(
                history_df, 
                x='timestamp', 
                y='confidence',
                color='emotion',
                title="🎭 Emotion Confidence Timeline",
                color_discrete_map={emotion: config['color'] for emotion, config in EMOTION_CONFIG.items()}
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🧠 <strong>Advanced Emotion Recognition Platform</strong></p>
        <p>💡 This tool provides AI-powered emotion analysis for educational and wellness purposes.</p>
        <p>⚠️ Not a substitute for professional mental health consultation.</p>
        <p>🔒 Your data is processed locally and not stored permanently.</p>
    </div>
""", unsafe_allow_html=True)
