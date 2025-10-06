import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_PATH = "credit_risk_dataset.csv"
MODEL_PATH = "best_model.joblib"
FEATURE_META_PATH = "feature_meta.json"
CV_RESULTS_PATH = "cv_results.json"
TEST_METRICS_PATH = "test_metrics.json"

# Toggle for lengthy operations
RUN_FULL_TRAIN = False  # Set to True for full cross-validation
SAMPLE_MAX_ROWS = None  # Set to an integer to limit data rows for quick testing, or None for all data

def load_data(data_path):
    """
    Load the dataset from the given path and display basic info.
    """
    print(f"📂 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"📊 Dataset shape: {df.shape} (rows, columns)")
    print("\n🔍 First 5 rows:")
    print(df.head().to_string())
    return df

def detect_and_prepare_target(df):
    # Common target column names for credit risk
    possible_targets = ["default", "loan_status", "target", "is_default", "loan_default", "defaulted", "status", "loan_status_flag"]
    target_col = None
    for t in possible_targets:
        for c in df.columns:
            if c.lower() == t:
                target_col = c
                break
        if target_col:
            break

    # Heuristic: binary column with 0/1 or yes/no
    if target_col is None:
        for c in df.columns:
            uniq = df[c].dropna().unique()
            uniq_str = set([str(x).lower() for x in uniq[:50]])
            if uniq_str.issubset({"0", "1", "yes", "no", "y", "n", "true", "false"}):
                target_col = c
                break

    if target_col is None:
        # Fallback: last column
        target_col = df.columns[-1]
        print("⚠️ Target not found with heuristics; using last column:", target_col)
    else:
        print(f"✅ Detected target column: {target_col}")

    # Normalize target to 0/1
    def normalize_target(series):
        """
        Convert various binary representations to 0/1.
        """
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors='coerce')
        s = series.astype(str).str.strip().str.lower()
        mapdict = {"yes":1,"y":1,"true":1,"t":1,"1":1,"no":0,"n":0,"false":0,"f":0,"0":0,"default":1,"charged off":1,"charged_off":1,"paid":0,"fully paid":0}
        return s.map(mapdict)

    y = normalize_target(df[target_col])
    # If too many NaNs, try numeric conversion
    if y.isnull().sum() > 0.5 * len(y):
        try:
            y = pd.to_numeric(df[target_col], errors='coerce')
        except:
            pass

    # Drop rows with missing target
    if y.isnull().any():
        print(f"🗑️ Dropping {y.isnull().sum()} rows with missing/ambiguous target")
        mask = ~y.isnull()
        df = df.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

    # Ensure target is integer 0/1
    y = y.astype(int)

    print(f"📈 Final sample size: {len(y)} | Defaults (1): {y.sum()} | Non-defaults (0): {len(y)-y.sum()}")
    return df, y, target_col

def perform_eda(df, y, target_col):
    """
    Perform exploratory data analysis, create visualizations, and save them.
    """
    print("\n🔍 --- Exploratory Data Analysis ---")
    print(df.info())

    # Missing values
    mv = df.isnull().sum().sort_values(ascending=False)
    print("\n❓ Top missing values:")
    print(mv.head(10).to_string())

    # Feature types
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Low-cardinality numeric as categorical
    for c in numeric_cols[:]:
        if X[c].nunique() <= 10 and not pd.api.types.is_float_dtype(X[c]):
            numeric_cols.remove(c)
            cat_cols.append(c)

    print(f"\n📊 Numeric features: {len(numeric_cols)} | Categorical features: {len(cat_cols)}")

    # Save sample snapshot
    df.sample(200, random_state=1).to_csv("sample_snapshot.csv", index=False)

    # Target distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x=y, palette="viridis")
    plt.title("Loan Default Distribution", fontsize=14)
    plt.xlabel("Default Status (0=No, 1=Yes)")
    plt.ylabel("Count")
    plt.savefig("target_distribution.png", bbox_inches='tight')
    plt.close()

    # Numeric summaries and histograms
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc.to_csv("numeric_describe.csv")

        # Histograms for key numeric features
        key_num = numeric_cols[:6]
        for col in key_num:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col].dropna(), bins=30, kde=True, color="skyblue")
            plt.title(f"Distribution of {col}", fontsize=14)
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.savefig(f"hist_{col}.png", bbox_inches='tight')
            plt.close()

        # Boxplots for numeric vs target
        for col in key_num[:4]:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=y, y=df[col], palette="Set2")
            plt.title(f"{col} by Default Status", fontsize=14)
            plt.xlabel("Default (0/1)")
            plt.ylabel(col)
            plt.savefig(f"box_{col}_vs_target.png", bbox_inches='tight')
            plt.close()

    # Categorical bar plots
    for c in cat_cols[:5]:
        plt.figure(figsize=(8,5))
        top_vals = df[c].fillna("MISSING").value_counts().head(10)
        sns.barplot(x=top_vals.values, y=top_vals.index, palette="coolwarm")
        plt.title(f"Top Values in {c}", fontsize=14)
        plt.xlabel("Count")
        plt.ylabel(c)
        plt.savefig(f"bar_{c}.png", bbox_inches='tight')
        plt.close()

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        num_to_corr = numeric_cols if len(numeric_cols) <= 15 else sorted(numeric_cols, key=lambda x: df[x].var(), reverse=True)[:15]
        corr = df[num_to_corr].corr()
        plt.figure(figsize=(12,10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap="RdYlBu", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap", fontsize=16)
        plt.savefig("correlation_heatmap.png", bbox_inches='tight')
        plt.close()

    # Pairplot for top numeric features
    if len(numeric_cols) >= 3:
        pair_cols = numeric_cols[:4] + [target_col]
        sns.pairplot(df[pair_cols], hue=target_col, palette="husl", diag_kind="kde")
        plt.savefig("pairplot_features.png", bbox_inches='tight')
        plt.close()

    print("✅ EDA visualizations saved in current directory.")
    return numeric_cols, cat_cols

# Main execution starts here
if __name__ == "__main__":
    # Step 1: Load data
    df = load_data(DATA_PATH)

    # Step 2: Detect and prepare target
    df, y, target_col = detect_and_prepare_target(df)

    # Step 3: Perform EDA
    numeric_cols, cat_cols = perform_eda(df, y, target_col)

    ########################
    # PREPROCESSING PIPELINE
    ########################

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split

    # Feature list
    FEATURES = numeric_cols + cat_cols

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'
    )

    ########################
    # FEATURE ENGINEERING
    ########################

    # Create domain-specific features for credit risk
    print("\n🔧 Engineering features...")

    if 'person_income' in df.columns and 'loan_amnt' in df.columns:
        df['loan_to_income'] = df['loan_amnt'] / (df['person_income'] + 1e-9)
        if 'loan_to_income' not in numeric_cols:
            numeric_cols.append('loan_to_income')
            FEATURES.append('loan_to_income')
        print("✅ Created loan_to_income ratio.")

    # Age groups for better interpretability
    if 'person_age' in df.columns:
        df['age_group'] = pd.cut(df['person_age'], bins=[0, 25, 35, 50, 100], labels=['Young', 'Adult', 'Middle', 'Senior'])
        if 'age_group' not in cat_cols:
            cat_cols.append('age_group')
            FEATURES.append('age_group')
        print("✅ Created age_group categorical feature.")

    # High income flag
    if 'person_income' in df.columns:
        df['high_income'] = (df['person_income'] > df['person_income'].median()).astype(int)
        if 'high_income' not in numeric_cols:
            numeric_cols.append('high_income')
            FEATURES.append('high_income')
        print("✅ Created high_income binary feature.")

    ########################
    # TRAIN/TEST SPLIT
    ########################

    # Sample for quick testing if specified
    if SAMPLE_MAX_ROWS is not None and len(df) > SAMPLE_MAX_ROWS:
        print(f"🧪 Sampling {SAMPLE_MAX_ROWS} rows for quick experimentation.")
        sample_idx = df.sample(SAMPLE_MAX_ROWS, random_state=42).index
        df = df.loc[sample_idx].reset_index(drop=True)
        y = y.loc[sample_idx].reset_index(drop=True)

    X = df[FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    print(f"🧠 Train/Test split: {X_train.shape} / {X_test.shape}")

    ########################
    # MODEL TRAINING & EVALUATION
    ########################

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_validate, StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # Define models with balanced classes for imbalanced data
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced'),
        "SVM": SVC(probability=True, class_weight='balanced', random_state=42)
    }

    # Add XGBoost if available
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
        print("✅ XGBoost available and added to models.")
    except ImportError:
        print("⚠️ XGBoost not installed. Install with `pip install xgboost` for better performance.")

    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}

    if RUN_FULL_TRAIN:
        print("🚀 Running full cross-validation (this may take a while)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, clf in models.items():
            print(f"Training {name} with CV...")
            pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
            scores = cross_validate(pipe, X_train, y_train, scoring=scoring_metrics, cv=cv, n_jobs=-1)
            cv_results[name] = {m: np.mean(scores[f'test_{m}']) for m in scoring_metrics}
        # Save CV results
        with open(CV_RESULTS_PATH, 'w') as f:
            json.dump(cv_results, f, indent=2)
    else:
        print("⚡ Quick training on train set and evaluation on test set...")
        for name, clf in models.items():
            print(f"Fitting {name}...")
            pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else None
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
            }
            cv_results[name] = {k: float(v) for k, v in metrics.items()}
        # Save test metrics
        with open(TEST_METRICS_PATH, 'w') as f:
            json.dump(cv_results, f, indent=2)

    print("📊 Model evaluation complete. Results saved.")

    ########################
    # SELECT BEST MODEL & SAVE
    ########################

    # Select best model prioritizing ROC-AUC for imbalanced classification
    best_model_name = max(cv_results, key=lambda k: cv_results[k].get('roc_auc', 0))
    best_score = cv_results[best_model_name].get('roc_auc', 0)
    print(f"🏆 Best model: {best_model_name} (ROC-AUC: {best_score:.4f})")

    best_clf = models[best_model_name]
    best_pipe = Pipeline([('preprocessor', preprocessor), ('clf', best_clf)])
    best_pipe.fit(X_train, y_train)
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"💾 Best model pipeline saved to {MODEL_PATH}")

    ########################
    # DETAILED EVALUATION & VISUALIZATIONS
    ########################

    from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report

    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_pipe, 'predict_proba') else None

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Default', 'Default'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues')
    plt.title(f"Confusion Matrix - {best_model_name}", fontsize=14)
    plt.savefig("confusion_matrix.png", bbox_inches='tight', dpi=300)
    plt.close()

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {best_model_name}', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig("roc_curve.png", bbox_inches='tight', dpi=300)
        plt.close()

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=['No Default', 'Default'], output_dict=True)
    with open("classification_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

    # Feature Importances (for tree-based models)
    if hasattr(best_clf, 'feature_importances_'):
        try:
            # Get feature names post-preprocessing
            preproc = best_pipe.named_steps['preprocessor']
            num_features = len(numeric_cols)
            ohe = preproc.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(cat_cols)
            all_feature_names = list(numeric_cols) + list(cat_feature_names)

            importances = best_clf.feature_importances_
            fi_df = pd.DataFrame({
                'feature': all_feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            fi_df.head(30).to_csv("feature_importances.csv", index=False)

            plt.figure(figsize=(10, 8))
            top_features = fi_df.head(20)
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title(f"Top 20 Feature Importances - {best_model_name}", fontsize=14)
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig("feature_importances.png", bbox_inches='tight', dpi=300)
            plt.close()

            print("\n🔍 Top 10 Important Features:")
            print(fi_df.head(10).to_string(index=False))
        except Exception as e:
            print(f"⚠️ Error in feature importance calculation: {e}")

    print("📈 Evaluation plots and reports saved in current directory.")

    ########################
    # FEATURE METADATA FOR UI
    ########################

    feature_metadata = {}
    for col in numeric_cols:
        series = df[col].dropna()
        feature_metadata[col] = {
            'type': 'numeric',
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': float(series.mean())
        }
    for col in cat_cols:
        unique_vals = df[col].dropna().unique().tolist()
        feature_metadata[col] = {
            'type': 'categorical',
            'choices': [str(v) for v in unique_vals[:100]]  # Limit for UI
        }

    with open(FEATURE_META_PATH, 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"📝 Feature metadata saved to {FEATURE_META_PATH}")

    ########################
    # SHAP EXPLAINABILITY (Optional)
    ########################

    try:
        import shap
        print("🧠 Computing SHAP explanations...")
        # Use preprocessed sample for SHAP
        X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        X_sample_processed = best_pipe.named_steps['preprocessor'].transform(X_sample)
        explainer = shap.Explainer(best_pipe.named_steps['clf'], X_sample_processed)
        shap_values = explainer(X_sample_processed)

        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample_processed, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance - {best_model_name}")
        plt.savefig("shap_summary.png", bbox_inches='tight', dpi=300)
        plt.close()

        # Waterfall for a sample prediction
        sample_idx = 0
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[sample_idx], show=False)
        plt.savefig("shap_waterfall_sample.png", bbox_inches='tight', dpi=300)
        plt.close()

        print("✅ SHAP plots saved.")
    except ImportError:
        print("⚠️ SHAP not installed. Run `pip install shap` to enable explainability.")
    except Exception as e:
        print(f"⚠️ SHAP computation failed: {e}")

    ########################
    # GENERATE README
    ########################

    readme_content = f"""# Credit Risk Prediction Project

This project builds a machine learning model to predict loan default risk using historical applicant data. It covers the full ML pipeline from data exploration to deployment-ready predictions.

## 🎯 Project Overview
- **Goal**: Classify loan applications as low-risk (0) or high-risk (1) for default.
- **Dataset**: Credit risk features like age, income, loan amount, employment length, etc.
- **Models**: Logistic Regression, Random Forest, XGBoost, SVM (best model: {best_model_name} with ROC-AUC {best_score:.3f}).
- **Key Insights**: {fi_df.head(3)['feature'].tolist() if 'fi_df' in locals() else 'Feature engineering and preprocessing improved model performance.'}

## 📊 Exploratory Data Analysis
Visualizations generated:
- Target distribution and class imbalance.
- Histograms and boxplots for numeric features.
- Bar charts for categorical features.
- Correlation heatmap and pairplots.

Check PNG files in the current directory for plots.

## 🧠 Model Performance
- **Test ROC-AUC**: {best_score:.3f}
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
{json.dumps(cv_results, indent=2)}

## 🔧 Customization
- Set `RUN_FULL_TRAIN = True` for 5-fold CV.
- Adjust `SAMPLE_MAX_ROWS` for faster testing.
- Add more features in the engineering section.

## 📚 Learning Outcomes
- Handling imbalanced datasets with class weights.
- End-to-end ML pipeline with scikit-learn.
- Model interpretability using SHAP.
- Deployment with Streamlit.

"""

    with open("README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("📄 Updated README.md with project summary and results.")

    print("\n🎉 Project complete! All outputs saved in the current directory.")
    print("Visualize results with the generated PNG files and JSON reports.")
