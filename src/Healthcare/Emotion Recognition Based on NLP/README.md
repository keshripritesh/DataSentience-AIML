# 🧠 Advanced Emotion Recognition Platform

A state-of-the-art emotion analysis platform powered by machine learning that provides real-time emotion detection with professional mental health insights.

![Python](https://img.shields.io/badge/Python-3.8+-brightgreen.svg) ![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg) ![scikit-learn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg) ![Plotly](https://img.shields.io/badge/Visualization-Plotly-blue.svg)

## ✨ Features

### 🎭 **Core Emotion Detection**
- **6 Emotion Classes**: Joy, Sadness, Anger, Fear, Love, Surprise
- **Smart Negation Handling**: Advanced logic for "I don't feel good" → Sadness
- **High Accuracy**: 88.58% accuracy with Random Forest model
- **Real-time Analysis**: Instant emotion detection as you type

### 📊 **Analysis Modes**
1. **Single Text Analysis**: Analyze individual texts with detailed insights
2. **Batch Processing**: Upload files or analyze multiple texts at once
3. **Real-time Stream**: Live emotion detection with dynamic visualizations

### 🎨 **Professional UI/UX**
- Healthcare-grade design with dynamic color animations
- Interactive gauges and confidence charts
- Responsive layout with hover effects
- Mental health insights and recommendations

### 📈 **Advanced Analytics**
- Emotion confidence scoring
- Historical emotion tracking
- Interactive Plotly visualizations
- Downloadable analysis results

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation & Usage
```bash
# Clone the repository
git clone https://github.com/SiyaDadpe/DataSentience-AIML.git

# Navigate to the project
cd "DataSentience-AIML/src/Healthcare/Emotion Recognition Based on NLP"

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### 🎯 **Test the Smart Model**
Try these phrases to see advanced negation handling:
- "I don't feel so good" → **Sadness** ✅
- "I'm not feeling well" → **Sadness** ✅
- "I don't like this" → **Anger** ✅
- "This is amazing" → **Joy** ✅

## 🧠 Model Architecture

- **Algorithm**: Random Forest Classifier (Smart Model)
- **Fallback**: Multinomial Naive Bayes (Basic Model)
- **Features**: TF-IDF Vectorization with n-grams
- **Dataset**: 20,000 emotion-labeled texts
- **Preprocessing**: Advanced negation detection and handling

## 📁 Project Structure
```
├── streamlit_app.py          # Main Streamlit application
├── train_smart_model.py      # Smart model training script
├── train_model.py           # Basic model training script
├── requirements.txt         # Dependencies
├── nlp_smart.pkl           # Smart emotion model
├── transform_smart.pkl     # Smart model vectorizer
├── nlp.pkl                # Basic emotion model (fallback)
├── transform.pkl          # Basic model vectorizer (fallback)
├── Dataset/               # Training data
├── app.py                # Legacy Flask app (for comparison)
└── README.md             # This file
```

## 🔬 Technical Details

### Smart Negation Handling
- Detects negation patterns: "not", "don't", "never", etc.
- Context-aware emotion adjustment
- Confidence-based overrides for better accuracy

### Mental Health Insights
- Emotion-specific wellness tips
- Professional recommendations
- Educational content (not medical advice)

## 🎨 UI Features
- **Dynamic Colors**: Confidence-based color intensity
- **Animations**: Pulsing effects and smooth transitions
- **Professional Design**: Healthcare-grade styling
- **Responsive Layout**: Works on all screen sizes

## ⚠️ Disclaimer
This tool is for educational and wellness purposes only. It is not a substitute for professional mental health consultation.

## 📄 License
This project is part of the DataSentience-AIML repository and follows the repository's licensing terms.

## 🤝 Contributing
Feel free to open issues and pull requests for improvements and bug fixes.

---
**Built with ❤️ using Streamlit, Plotly, and scikit-learn**
