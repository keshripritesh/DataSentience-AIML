# Credit Risk Prediction Project

This project builds a machine learning model to predict loan default risk using historical applicant data. It covers the full ML pipeline from data exploration to deployment-ready predictions.

## 🎯 Project Overview
- **Goal**: Classify loan applications as low-risk (0) or high-risk (1) for default.
- **Dataset**: Credit risk features like age, income, loan amount, employment length, etc.
- **Models**: Logistic Regression, Random Forest, XGBoost, SVM (best model: XGBoost with ROC-AUC 0.953).
- **Key Insights**: ['person_home_ownership_RENT', 'loan_to_income', 'loan_grade_C']

## 📊 Exploratory Data Analysis
Visualizations generated:
- Target distribution and class imbalance.
- Histograms and boxplots for numeric features.
- Bar charts for categorical features.
- Correlation heatmap and pairplots.

Check PNG files in the current directory for plots.

## 🧠 Model Performance
- **Test ROC-AUC**: 0.953
- **Precision/Recall/F1**: See `classification_report.json`.
- **Confusion Matrix**: `confusion_matrix.png`

## 🌐 Deployment
- Run `streamlit run streamlit_app.py` for an interactive prediction app.
- Input applicant details to get risk scores and explanations.

## 🚀 Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run this script: `python credit_risk_project_notebook.py`
3. Launch app: `streamlit run streamlit_app.py`

## 📈 Results Summary
{
  "Logistic Regression": {
    "accuracy": 0.8195488721804511,
    "precision": 0.5617469879518072,
    "recall": 0.7869198312236287,
    "f1": 0.655536028119508,
    "roc_auc": 0.8790598874548143
  },
  "Random Forest": {
    "accuracy": 0.9326377167408316,
    "precision": 0.9685414680648237,
    "recall": 0.7144866385372715,
    "f1": 0.8223391339538648,
    "roc_auc": 0.9292545020144676
  },
  "SVM": {
    "accuracy": 0.8777044652447445,
    "precision": 0.7012234385061172,
    "recall": 0.7658227848101266,
    "f1": 0.7321008403361344,
    "roc_auc": 0.9092526800909305
  },
  "XGBoost": {
    "accuracy": 0.9349393892895504,
    "precision": 0.9577981651376147,
    "recall": 0.7341772151898734,
    "f1": 0.8312101910828026,
    "roc_auc": 0.9531956124768637
  }
}

## 🔧 Customization
- Set `RUN_FULL_TRAIN = True` for 5-fold CV.
- Adjust `SAMPLE_MAX_ROWS` for faster testing.
- Add more features in the engineering section.

## 📚 Learning Outcomes
- Handling imbalanced datasets with class weights.
- End-to-end ML pipeline with scikit-learn.
- Model interpretability using SHAP.
- Deployment with Streamlit.

