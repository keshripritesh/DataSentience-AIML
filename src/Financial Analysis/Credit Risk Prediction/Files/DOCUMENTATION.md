# Credit Risk Prediction Project - Full Documentation

This document provides a comprehensive overview of the Credit Risk Prediction project. It covers the project's purpose, methodology, implementation details, deployment, and usage instructions. The project uses machine learning to predict loan default risk based on applicant data, following a complete end-to-end pipeline from data loading to interactive deployment.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Setup and Installation](#setup-and-installation)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Deployment with Streamlit](#deployment-with-streamlit)
- [Usage Instructions](#usage-instructions)
- [Results and Insights](#results-and-insights)
- [Generated Artifacts](#generated-artifacts)
- [Customization and Extensions](#customization-and-extensions)
- [Troubleshooting](#troubleshooting)
- [Learning Outcomes](#learning-outcomes)
- [Future Improvements](#future-improvements)
- [License and Credits](#license-and-credits)

## Project Overview
### Goal
The primary objective is to build a machine learning model that classifies loan applications as low-risk (0: no default) or high-risk (1: default) based on historical applicant data. This helps financial institutions assess credit risk more accurately, reducing potential losses from defaults.

### Key Components
- **Data Loading and Preparation**: Automatic detection and normalization of the target variable (loan default status).
- **Exploratory Data Analysis (EDA)**: Visualizations to understand data distribution, correlations, and patterns.
- **Preprocessing Pipeline**: Handles missing values, scaling, and encoding using scikit-learn.
- **Feature Engineering**: Domain-specific features like loan-to-income ratio and age groups.
- **Model Training**: Compares multiple classifiers (Logistic Regression, Random Forest, SVM, XGBoost) with cross-validation or train-test split.
- **Evaluation**: Metrics like ROC-AUC, precision, recall, F1-score; visualizations including confusion matrix and ROC curve.
- **Deployment**: Interactive Streamlit app for single and batch predictions with explainability.
- **Explainability**: SHAP values for model interpretability (if SHAP is installed).

### Technologies Used
- **Python**: Core language (version 3.8+ recommended).
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, Streamlit, XGBoost (optional), SHAP (optional).
- **Tools**: Jupyter/Notebook compatible script for reproducibility.

The best model typically achieves a ROC-AUC score of ~0.95, indicating strong performance on imbalanced credit risk data.

## Dataset Description
The dataset (`credit_risk_dataset.csv`) contains historical loan applicant information. Key features include:

### Target Variable
- **Default Status** (binary: 0 = No Default, 1 = Default): The label to predict. Automatically detected from common column names (e.g., "default", "loan_status") or heuristics (binary-like columns).

### Features
- **Numeric Features** (examples):
  - `person_age`: Applicant's age (years).
  - `person_income`: Annual income ($).
  - `loan_amnt`: Requested loan amount ($).
  - `loan_int_rate`: Interest rate (%).
  - `person_emp_length`: Employment length (years).
  - `loan_percent_income`: Loan as % of income.
  - `cb_person_cred_hist_length`: Credit history length (years).

- **Categorical Features** (examples):
  - `person_home_ownership`: RENT, OWN, MORTGAGE, OTHER.
  - `loan_intent`: PERSONAL, EDUCATION, MEDICAL, VENTURE, etc.
  - `loan_grade`: A, B, C, D, E, F, G (credit grade).
  - `cb_person_default_on_file`: Y/N (previous default).

### Data Characteristics
- **Size**: ~32,000 rows, 12-15 columns (varies by exact dataset).
- **Imbalance**: Defaults are typically a minority class (~20-30%), handled via class weights.
- **Missing Values**: Imputed (median for numeric, most frequent for categorical).
- **Sample Snapshot**: A CSV sample (`sample_snapshot.csv`) is generated for quick inspection.

For privacy, no real personal data is used; this is synthetic or anonymized for demonstration.

## Setup and Installation
1. **Clone/Navigate to Project Directory**

2. **Install Dependencies**:
   Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```
   Key packages:
   - `pandas`, `numpy`: Data manipulation.
   - `scikit-learn`: ML pipeline.
   - `matplotlib`, `seaborn`: Visualizations.
   - `joblib`: Model serialization.
   - `streamlit`: Deployment.
   - `xgboost`: Advanced model (optional: `pip install xgboost`).
   - `shap`: Explainability (optional: `pip install shap`).

3. **Verify Setup**:
   Run `python -c "import pandas, sklearn, streamlit; print('Setup OK')"` to check imports.

## Project Structure
```
credit_risk_project/
├── credit_risk_dataset.csv          # Input dataset
├── requirements.txt                 # Dependencies
├── credit_risk_project_notebook.py  # Main script: EDA, training, evaluation
├── streamlit_app.py                 # Deployment app
├── README.md                        # Quick overview
├── DOCUMENTATION.md                 # This full documentation
├── best_model.joblib                # Trained model pipeline
├── feature_meta.json                # UI metadata for features
├── test_metrics.json                # Model performance
├── classification_report.json       # Detailed metrics
├── feature_importances.csv          # Top features
├── sample_snapshot.csv              # Data preview
├── numeric_describe.csv             # Numeric summaries
├── *.png                            # Visualizations (histograms, heatmaps, etc.)
└── cv_results.json                  # Cross-validation results (if enabled)
```

## Exploratory Data Analysis (EDA)
EDA is performed in `credit_risk_project_notebook.py` to uncover data insights.

### Steps
1. **Data Loading**: Loads CSV and displays shape, head, and info.
2. **Target Detection**: Auto-detects binary target column.
3. **Missing Values**: Reports top missing columns.
4. **Feature Typing**: Separates numeric and categorical features (low-cardinality numerics treated as categorical).

### Visualizations Generated
- **Target Distribution** (`target_distribution.png`): Bar plot showing class imbalance.
- **Histograms** (`hist_*.png`): Distributions for key numeric features (e.g., age, income).
- **Boxplots** (`box_*.vs_target.png`): Numeric features vs. default status to spot differences.
- **Bar Charts** (`bar_*.png`): Top values for categorical features (e.g., loan grade, home ownership).
- **Correlation Heatmap** (`correlation_heatmap.png`): Pearson correlations for numeric features.
- **Pairplot** (`pairplot_features.png`): Scatter plots with hue by target for top features.

### Outputs
- `numeric_describe.csv`: Summary statistics (mean, std, min/max) for numerics.
- Console prints: Dataset info, missing values, feature counts.

Run the script with `python credit_risk_project_notebook.py` to generate these.

## Feature Engineering
Custom features are added to improve model performance, based on credit risk domain knowledge.

### Engineered Features
1. **Loan-to-Income Ratio** (`loan_to_income`): `loan_amnt / person_income` (numeric). High ratios indicate financial strain.
2. **Age Group** (`age_group`): Binned from `person_age` (categorical): Young (0-24), Adult (25-34), Middle (35-49), Senior (50+).
3. **High Income Flag** (`high_income`): Binary (1 if income > median, else 0).

These are created before train-test split and included in the preprocessing pipeline.

### Preprocessing Pipeline
- **Numeric**: Impute median → StandardScaler.
- **Categorical**: Impute most frequent → OneHotEncoder (handles unknown categories).
- **ColumnTransformer**: Applies transformations selectively.
- **Metadata**: `feature_meta.json` saves ranges/choices for UI inputs.

## Model Training and Evaluation
Training uses scikit-learn pipelines for reproducibility.

### Models Compared
- **Logistic Regression**: Linear baseline with class weights for imbalance.
- **Random Forest**: Ensemble tree method.
- **SVM**: Support Vector Machine with probability estimates.
- **XGBoost** (if installed): Gradient boosting, often the best performer.

### Training Modes
- **Quick Mode** (default, `RUN_FULL_TRAIN=False`): Train on 80% data, evaluate on 20% test set.
- **Full Mode** (`RUN_FULL_TRAIN=True`): 5-fold Stratified K-Fold CV (slower, more robust).

### Metrics
- Primary: ROC-AUC (handles imbalance well).
- Others: Accuracy, Precision, Recall, F1-Score.
- Best model selected by highest ROC-AUC and saved as `best_model.joblib`.

### Evaluation Outputs
- **Confusion Matrix** (`confusion_matrix.png`): True vs. predicted labels.
- **ROC Curve** (`roc_curve.png`): TPR vs. FPR with AUC.
- **Classification Report** (`classification_report.json`): Per-class metrics.
- **Feature Importances** (`feature_importances.csv/png`): Top features for tree models (e.g., loan_to_income, home_ownership_RENT).
- **Test Metrics** (`test_metrics.json`): Dict of model performances.
- **SHAP Plots** (if installed): `shap_summary.png` (global importance), `shap_waterfall_sample.png` (local explanation).

Console prints top features and metrics.

## Deployment with Streamlit
The `streamlit_app.py` provides an interactive web interface.

### App Features
- **Home**: Project overview, key metrics (ROC-AUC, accuracy, etc.).
- **Single Prediction**: Form for applicant inputs (sliders, dropdowns); outputs risk level, probability bar, recommendations, key factors.
- **Model Performance**: Model comparison table, ROC-AUC bar chart, precision-recall scatter.
- **Feature Analysis**: Top 20 importances bar plot, feature explanations.
- **Batch Prediction**: Upload CSV, get predictions, summary stats, pie chart, download results.

### Styling and UX
- Custom CSS: Colored risk boxes (green for low, red for high), metric cards.
- Engineered features auto-calculated (e.g., loan-to-income).
- Error handling: Model/data checks, input validation.
- Sidebar: Navigation and support info.

### Screenshots
See `interface.png` for a sample view of the dashboard.

## Usage Instructions
1. **Run Training/EDA**:
   ```
   python credit_risk_project_notebook.py
   ```
   - Generates all artifacts (models, plots, JSONs).
   - Edit `RUN_FULL_TRAIN` or `SAMPLE_MAX_ROWS` for customization.

2. **Launch App**:
   ```
   streamlit run streamlit_app.py
   ```
   - Opens in browser (default: http://localhost:8501).
   - Use navigation sidebar to switch pages.

3. **Batch Prediction**:
   - Prepare CSV with feature columns (matching training).
   - Upload in app; download results CSV with predictions.

4. **View Results**:
   - Open PNG files in any image viewer.
   - JSONs in text editor or `pd.read_json()` in Python.

## Results and Insights
### Typical Performance (XGBoost)
- ROC-AUC: 0.953
- Accuracy: 0.935
- Precision: 0.958 (low false positives)
- Recall: 0.734 (catches most defaults)
- F1: 0.831

### Key Insights
- **Top Risk Factors**: High loan-to-income, renting (vs. owning), lower credit grades (C/D), previous defaults.
- **Imbalance Handling**: Class weights prevent bias toward majority class.
- **Feature Impact**: Engineered features like `loan_to_income` rank highly in importances.
- **SHAP Insights**: Renters show higher default risk; medical/debt consolidation intents increase probability.

Full results in `test_metrics.json` and plots.

## Generated Artifacts
- **Models**: `best_model.joblib` (load with `joblib.load()`).
- **Metrics**: `test_metrics.json`, `classification_report.json`, `cv_results.json`.
- **Features**: `feature_importances.csv`, `feature_meta.json`.
- **Data**: `sample_snapshot.csv`, `numeric_describe.csv`.
- **Visuals**: 20+ PNGs (histograms, boxplots, heatmaps, ROC, SHAP, etc.).

## Customization and Extensions
- **Toggle Flags** (in notebook):
  - `RUN_FULL_TRAIN=True`: Enable CV.
  - `SAMPLE_MAX_ROWS=1000`: Faster testing on subset.
- **Add Models**: Extend `models` dict in notebook.
- **New Features**: Add in engineering section; update pipeline.
- **Threshold Tuning**: In app, adjust prediction threshold for precision/recall trade-off.
- **API Deployment**: Wrap model in FastAPI for production.

## Troubleshooting
- **Model Not Found**: Run notebook first to generate `best_model.joblib`.
- **Import Errors**: Check `requirements.txt`; install XGBoost/SHAP separately if needed.
- **Streamlit Issues**: Ensure port 8501 free; restart with `streamlit run streamlit_app.py --server.port 8502`.
- **Data Errors**: Verify CSV has expected columns; target detection may need manual override.
- **SHAP Fails**: Install `pip install shap`; or skip (non-critical).
- **Windows Paths**: Use forward slashes or `Path` objects.

## Learning Outcomes
- **ML Pipeline**: End-to-end with scikit-learn (preprocessing, training, evaluation).
- **Imbalanced Data**: Class weights, ROC-AUC for binary classification.
- **Feature Engineering**: Domain knowledge improves performance.
- **Interpretability**: SHAP for "why" behind predictions.
- **Deployment**: Streamlit for quick, interactive apps.
- **Reproducibility**: Seeds, pipelines, artifact saving.

## Future Improvements
- **Hyperparameter Tuning**: Use GridSearchCV or Optuna.
- **Advanced Models**: Neural networks (TensorFlow/PyTorch) or ensemble stacking.
- **Data Augmentation**: SMOTE for oversampling minorities.
- **Monitoring**: Add logging, drift detection for production.
- **Security**: Input validation, API keys for real deployment.
- **More Visuals**: Interactive plots with Plotly in Streamlit.
- **Database Integration**: Load from SQL instead of CSV.

## License and Credits
- **License**: MIT (open-source; modify freely).
- **Credits**: Built with scikit-learn, Streamlit, XGBoost. Dataset: Synthetic credit risk (inspired by LendingClub/Kaggle).
- **Author**: [Anu] - For questions, review this doc or README.md.

---

