# 🌌 NASA Asteroids Dataset Analysis

This project analyzes NASA’s asteroid dataset to uncover key insights and predict asteroid hazardness. It combines **data cleaning, visualization, feature engineering, and machine learning (XGBoost)**, with a **Streamlit app** for real-time predictions.

## 📊 Workflow
- **Data Cleaning**: Removed duplicates, irrelevant features, and handled missing values.  
- **EDA**: Visualized distributions, correlations, and asteroid characteristics.  
- **Feature Engineering**: Focused on quantitative features affecting hazardness.  
- **Machine Learning**: Trained an XGBoost classifier, saved as `xgb_model.pkl`.  
- **Deployment**: Streamlit app (`app.py`) for interactive predictions.  

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/Nasa-Asteroids-Dataset.git
cd Nasa-Asteroids-Dataset
pip install -r requirements.txt
streamlit run app.py
