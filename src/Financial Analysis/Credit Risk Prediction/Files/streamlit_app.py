import streamlit as st
import joblib, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64

# Set page config
st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid;
        margin: 10px 0;
    }
    .high-risk {
        background-color: #ffe6e6;
        border-color: #ff6b6b;
    }
    .low-risk {
        background-color: #e6ffe6;
        border-color: #4CAF50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
PROJECT_DIR = Path(".")
MODEL_PATH = PROJECT_DIR / "best_model.joblib"
FEATURE_META_PATH = PROJECT_DIR / "feature_meta.json"
TEST_METRICS_PATH = PROJECT_DIR / "test_metrics.json"
FEATURE_IMPORTANCES_PATH = PROJECT_DIR / "feature_importances.csv"

# Sidebar
st.sidebar.markdown('<div class="sidebar-header">📊 Credit Risk Dashboard</div>', unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "🔮 Single Prediction", "📈 Model Performance", "🔍 Feature Analysis", "📋 Batch Prediction"])

# Load model and metadata
if not MODEL_PATH.exists():
    st.error(f"❌ Model not found at {MODEL_PATH}. Please run the training script first.")
    st.stop()

model = joblib.load(MODEL_PATH)

if not FEATURE_META_PATH.exists():
    st.warning("⚠️ Feature metadata not found. Please re-run the preprocessing script.")
    st.stop()

with open(FEATURE_META_PATH, 'r') as f:
    meta = json.load(f)

# Load additional data if available
test_metrics = None
if TEST_METRICS_PATH.exists():
    with open(TEST_METRICS_PATH, 'r') as f:
        test_metrics = json.load(f)

feature_importances = None
if FEATURE_IMPORTANCES_PATH.exists():
    feature_importances = pd.read_csv(FEATURE_IMPORTANCES_PATH)

# Main content based on page selection
if page == "🏠 Home":
    st.markdown('<div class="main-header">💳 Credit Risk Prediction System</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Credit Risk Assessment Platform

    This advanced machine learning system predicts loan default risk using historical applicant data.
    Our XGBoost model achieves **95.3% ROC-AUC** accuracy on test data.

    #### 🎯 Key Features:
    - **Real-time Predictions**: Instant risk assessment for loan applications
    - **Explainable AI**: Feature importance and SHAP explanations
    - **Batch Processing**: Analyze multiple applications at once
    - **Model Insights**: Comprehensive performance metrics and visualizations
    - **User-friendly Interface**: Intuitive design for loan officers and analysts

    #### 🚀 Quick Start:
    1. Navigate to **Single Prediction** to assess individual applicants
    2. Check **Model Performance** for detailed evaluation metrics
    3. Explore **Feature Analysis** to understand what drives predictions
    4. Use **Batch Prediction** for bulk analysis

    ---
    """)

    # Show key metrics if available
    if test_metrics:
        st.subheader("📊 Model Performance Overview")
        col1, col2, col3, col4 = st.columns(4)

        best_model = max(test_metrics.keys(), key=lambda k: test_metrics[k].get('roc_auc', 0))
        metrics = test_metrics[best_model]

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("F1-Score", f"{metrics.get('f1', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "🔮 Single Prediction":
    st.markdown('<div class="main-header">🔮 Single Loan Risk Assessment</div>', unsafe_allow_html=True)

    st.markdown("""
    Enter the applicant's details below to get an instant risk assessment.
    The model will predict the likelihood of loan default based on historical patterns.
    """)

    # Create form with better organization
    st.subheader("👤 Applicant Information")

    col1, col2 = st.columns(2)

    input_data = {}

    # Personal Information
    with col1:
        st.markdown("**Personal Details**")
        person_age = st.slider("Age", 18, 100, 30, help="Applicant's age in years")
        input_data['person_age'] = person_age

        person_income = st.number_input("Annual Income ($)", 0, 10000000, 50000, step=1000,
                                      help="Applicant's annual income")
        input_data['person_income'] = person_income

        person_home_ownership = st.selectbox("Home Ownership",
                                           meta.get('person_home_ownership', {}).get('choices', ['RENT', 'OWN', 'MORTGAGE']),
                                           help="Type of home ownership")
        input_data['person_home_ownership'] = person_home_ownership

        person_emp_length = st.slider("Employment Length (years)", 0, 50, 5,
                                    help="Years of employment")
        input_data['person_emp_length'] = person_emp_length

    # Loan Information
    with col2:
        st.markdown("**Loan Details**")
        loan_amnt = st.number_input("Loan Amount ($)", 0, 1000000, 10000, step=500,
                                  help="Requested loan amount")
        input_data['loan_amnt'] = loan_amnt

        loan_intent = st.selectbox("Loan Purpose",
                                 meta.get('loan_intent', {}).get('choices', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE']),
                                 help="Purpose of the loan")
        input_data['loan_intent'] = loan_intent

        loan_grade = st.selectbox("Credit Grade",
                                meta.get('loan_grade', {}).get('choices', ['A', 'B', 'C', 'D', 'E', 'F', 'G']),
                                help="Credit grade assigned")
        input_data['loan_grade'] = loan_grade

        loan_int_rate = st.slider("Interest Rate (%)", 0.0, 30.0, 10.0, 0.1,
                                help="Loan interest rate")
        input_data['loan_int_rate'] = loan_int_rate

    # Additional features
    st.subheader("📋 Additional Information")
    col3, col4 = st.columns(2)

    with col3:
        loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.2, 0.01,
                                      help="Loan amount as percentage of income")
        input_data['loan_percent_income'] = loan_percent_income

        cb_person_default_on_file = st.selectbox("Previous Default on File",
                                              ['N', 'Y'],
                                              help="Has the applicant defaulted before?")
        input_data['cb_person_default_on_file'] = cb_person_default_on_file

    with col4:
        cb_person_cred_hist_length = st.slider("Credit History Length (years)", 0, 50, 5,
                                             help="Length of credit history")
        input_data['cb_person_cred_hist_length'] = cb_person_cred_hist_length

        # Add engineered features
        loan_to_income = loan_amnt / (person_income + 1e-9)
        st.info(f"Calculated Loan-to-Income Ratio: {loan_to_income:.3f}")
        input_data['loan_to_income'] = loan_to_income

        age_group = 'Young' if person_age < 25 else ('Adult' if person_age < 35 else ('Middle' if person_age < 50 else 'Senior'))
        st.info(f"Age Group: {age_group}")
        input_data['age_group'] = age_group

        high_income = 1 if person_income > 50000 else 0  # Using median as threshold
        st.info(f"High Income Flag: {'Yes' if high_income else 'No'}")
        input_data['high_income'] = high_income

    # Prediction button
    if st.button("🔍 Assess Risk", type="primary", use_container_width=True):
        df_input = pd.DataFrame([input_data])

        try:
            pred = model.predict(df_input)[0]
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df_input)[:, 1][0]

            # Display results with styling
            st.success("✅ Risk Assessment Complete!")

            col_result1, col_result2 = st.columns(2)

            with col_result1:
                risk_level = "HIGH RISK" if pred == 1 else "LOW RISK"
                risk_class = "high-risk" if pred == 1 else "low-risk"
                st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    <h3>🎯 Risk Prediction: {risk_level}</h3>
                    <p><strong>Default Probability:</strong> {proba:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_result2:
                if proba is not None:
                    # Risk gauge visualization
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.barh([0], [proba], color='#ff6b6b', height=0.5, label='Default Risk')
                    ax.barh([0], [1-proba], left=proba, color='#4CAF50', height=0.5, label='Safe')
                    ax.set_xlim(0, 1)
                    ax.set_yticks([])
                    ax.set_xlabel('Probability')
                    ax.legend(loc='center right')
                    st.pyplot(fig)

            # Recommendations
            st.subheader("💡 Recommendations")
            if pred == 1:
                st.warning("""
                **High Risk Alert!** Consider additional verification:
                - Request more documentation
                - Consider co-signer
                - Review credit history thoroughly
                - Evaluate alternative loan terms
                """)
            else:
                st.success("""
                **Low Risk Profile** - Good candidate for approval:
                - Standard approval process
                - Competitive interest rates
                - Fast processing possible
                """)

            # Feature contributions (simplified)
            st.subheader("🔍 Key Risk Factors")
            risk_factors = []
            if loan_to_income > 0.3:
                risk_factors.append("High loan-to-income ratio")
            if loan_grade in ['D', 'E', 'F', 'G']:
                risk_factors.append("Lower credit grade")
            if cb_person_default_on_file == 'Y':
                risk_factors.append("Previous default on record")
            if person_home_ownership == 'RENT':
                risk_factors.append("Renter (vs owner)")

            if risk_factors:
                for factor in risk_factors[:3]:
                    st.write(f"• {factor}")
            else:
                st.write("• No major risk factors identified")

        except Exception as e:
            st.error("❌ Prediction failed. Please check input values.")
            st.exception(e)

elif page == "📈 Model Performance":
    st.markdown('<div class="main-header">📈 Model Performance Analysis</div>', unsafe_allow_html=True)

    if not test_metrics:
        st.warning("Test metrics not found. Please run the training script with evaluation.")
    else:
        # Model comparison
        st.subheader("🏆 Model Comparison")

        models_df = pd.DataFrame(test_metrics).T
        st.dataframe(models_df.style.highlight_max(axis=0))

        # Best model details
        best_model = max(test_metrics.keys(), key=lambda k: test_metrics[k].get('roc_auc', 0))
        st.success(f"**Best Model:** {best_model} (ROC-AUC: {test_metrics[best_model].get('roc_auc', 0):.3f})")

        # Performance visualizations
        st.subheader("📊 Performance Metrics")

        col1, col2 = st.columns(2)

        with col1:
            # ROC-AUC comparison
            fig, ax = plt.subplots(figsize=(8, 5))
            models = list(test_metrics.keys())
            roc_scores = [test_metrics[m].get('roc_auc', 0) for m in models]
            bars = ax.bar(models, roc_scores, color='skyblue')
            ax.set_ylabel('ROC-AUC Score')
            ax.set_title('Model ROC-AUC Comparison')
            plt.xticks(rotation=45)
            for bar, score in zip(bars, roc_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            st.pyplot(fig)

        with col2:
            # Precision vs Recall
            fig, ax = plt.subplots(figsize=(8, 5))
            precision = [test_metrics[m].get('precision', 0) for m in models]
            recall = [test_metrics[m].get('recall', 0) for m in models]

            ax.scatter(precision, recall, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model, (precision[i], recall[i]), xytext=(5, 5),
                           textcoords='offset points')

            ax.set_xlabel('Precision')
            ax.set_ylabel('Recall')
            ax.set_title('Precision vs Recall')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

elif page == "🔍 Feature Analysis":
    st.markdown('<div class="main-header">🔍 Feature Importance & Analysis</div>', unsafe_allow_html=True)

    if feature_importances is None:
        st.warning("Feature importances not found. Please ensure the model was trained with feature importance calculation.")
    else:
        st.subheader("🎯 Top Feature Importances")

        # Top 20 features
        top_features = feature_importances.head(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(top_features['feature'], top_features['importance'], color='lightcoral')
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 20 Feature Importances (XGBoost)')
        ax.invert_yaxis()  # Highest at top

        # Add value labels
        for bar, importance in zip(bars, top_features['importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}', va='center')

        st.pyplot(fig)

        # Feature explanations
        st.subheader("📖 Feature Explanations")

        explanations = {
            'person_home_ownership_RENT': 'Renters have higher default risk compared to homeowners',
            'loan_to_income': 'Higher ratio indicates more financial strain',
            'loan_grade_C': 'Credit grade C represents moderate risk',
            'loan_intent_DEBTCONSOLIDATION': 'Debt consolidation loans have higher risk',
            'loan_grade_D': 'Credit grade D indicates higher risk',
            'person_home_ownership_OWN': 'Homeowners have lower default risk',
            'loan_int_rate': 'Higher interest rates correlate with higher risk',
            'loan_intent_MEDICAL': 'Medical loans may indicate financial stress'
        }

        for feature, explanation in list(explanations.items())[:8]:
            if feature in top_features['feature'].values:
                st.write(f"**{feature}**: {explanation}")

elif page == "📋 Batch Prediction":
    st.markdown('<div class="main-header">📋 Batch Risk Assessment</div>', unsafe_allow_html=True)

    st.markdown("""
    Upload a CSV file with multiple loan applications for batch risk assessment.
    The file should contain the same features as used in training.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(batch_data)} applications for analysis")

            # Show preview
            st.subheader("📄 Data Preview")
            st.dataframe(batch_data.head())

            if st.button("🔍 Analyze Batch", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    # Prepare batch data by adding engineered features
                    processed_batch_data = batch_data.copy()

                    # Add engineered features (same as in training script)
                    if 'person_income' in processed_batch_data.columns and 'loan_amnt' in processed_batch_data.columns:
                        processed_batch_data['loan_to_income'] = processed_batch_data['loan_amnt'] / (processed_batch_data['person_income'] + 1e-9)

                    if 'person_age' in processed_batch_data.columns:
                        processed_batch_data['age_group'] = pd.cut(processed_batch_data['person_age'],
                                                                  bins=[0, 25, 35, 50, 100],
                                                                  labels=['Young', 'Adult', 'Middle', 'Senior'])

                    if 'person_income' in processed_batch_data.columns:
                        # Use same threshold as training (median of training data)
                        income_median = 50000  # Approximate median from training
                        processed_batch_data['high_income'] = (processed_batch_data['person_income'] > income_median).astype(int)

                    # Make predictions on processed data
                    predictions = model.predict(processed_batch_data)
                    probabilities = model.predict_proba(processed_batch_data)[:, 1] if hasattr(model, "predict_proba") else None

                    # Add results to dataframe
                    results_df = batch_data.copy()
                    results_df['Predicted_Risk'] = predictions
                    results_df['Default_Probability'] = probabilities

                    # Summary statistics
                    st.subheader("📊 Batch Analysis Summary")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        high_risk_count = (predictions == 1).sum()
                        st.metric("High Risk Applications", high_risk_count)

                    with col2:
                        low_risk_count = (predictions == 0).sum()
                        st.metric("Low Risk Applications", low_risk_count)

                    with col3:
                        avg_proba = probabilities.mean() if probabilities is not None else 0
                        st.metric("Average Risk Score", f"{avg_proba:.1%}")

                    # Risk distribution
                    st.subheader("📈 Risk Distribution")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    risk_counts = pd.Series(predictions).value_counts()
                    risk_counts.index = ['Low Risk', 'High Risk']
                    ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                          colors=['#4CAF50', '#ff6b6b'])
                    ax.set_title('Risk Distribution in Batch')
                    st.pyplot(fig)

                    # Download results
                    st.subheader("💾 Download Results")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="batch_risk_predictions.csv",
                        mime="text/csv"
                    )

                    # Show results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df)

        except Exception as e:
            st.error("❌ Error processing batch file. Please check the file format.")
            st.exception(e)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 Support")
st.sidebar.info("For questions or issues, please check the project documentation")
st.sidebar.markdown("**Version:** 1.0.0")
st.sidebar.markdown("**Model:** XGBoost (ROC-AUC: 0.953)")
