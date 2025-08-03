# 🔮 Best GenAI Tool Predictor

This project aims to predict the most suitable Generative AI tool (e.g., **Gemini, Claude, Groq, Mixtral, LLaMA**) for a company based on its industry, location, workforce impact metrics, and employee sentiment. The prediction leverages structured company data along with NLP-based sentiment embeddings.

[!ui](assets/image.png)
---

## 📁 Project Structure

GenAI-Tool-Predictor/
│
├── data/
│ └── genai_tool_data.csv # Input dataset
│
├── model/
│ ├── genai_model.pkl # Trained classification model
│ ├── encoder.pkl # OneHotEncoder for categorical vars
│ ├── scaler.pkl # StandardScaler for numeric vars
│ └── vectorizer.pkl # TF-IDF Vectorizer for sentiment
│
├── preprocess.py # Preprocessing & embedding
├── train.py # Model training and evaluation
├── predict.py # Inference for new samples
└── README.md # Project overview


---

## 🧠 Problem Statement

Companies across the world are adopting GenAI tools. Each industry may benefit differently based on:

- Domain-specific needs
- Employee training and reaction
- Country-specific constraints
- Workforce size and transformation
- Sentiment and productivity outcomes

**Goal**: Predict the most suitable GenAI Tool a company should adopt using historical adoption patterns and company-specific data.

---

## 🎯 Target Variable

**`GenAI Tool`** (Categorical)

Example classes:
- Gemini
- Claude
- Groq
- Mixtral
- LLaMA

---

## 📊 Input Features

| Feature Name                     | Type        | Description |
|----------------------------------|-------------|-------------|
| Industry                         | Categorical | Business domain (e.g., Healthcare, Telecom) |
| Country                          | Categorical | Country where the company operates |
| Adoption Year                    | Numerical   | Year when AI was adopted |
| Number of Employees Impacted     | Numerical   | Number of employees affected by AI |
| New Roles Created                | Numerical   | Number of new roles post-AI |
| Training Hours Provided          | Numerical   | Hours of training given |
| Productivity Change (%)          | Numerical   | Improvement in productivity |
| Employee Sentiment               | Text        | Employee feedback on GenAI adoption |

---

## ⚙️ How It Works

### Preprocessing (`preprocess.py`)

- Drops irrelevant fields (`Company Name`)
- One-hot encodes `Industry` and `Country`
- Normalizes numerical features
- Uses **TF-IDF embedding** for `Employee Sentiment`
- Combines all into a final feature vector

### Model Training (`train.py`)

- Loads and preprocesses the data
- Trains a **Random Forest classifier**
- Saves the model and vectorizers to `model/`

### Prediction (`predict.py`)

- Loads a sample input dictionary
- Applies the same preprocessing logic
- Uses the trained model to predict the GenAI Tool

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/genai-tool-predictor.git
cd genai-tool-predictor
pip install -r requirements.txt
pandas
scikit-learn
joblib
numpy
python train.py
python predict.py


sample = {
    "Industry": "Healthcare",
    "Country": "USA",
    "Adoption Year": 2023,
    "Number of Employees Impacted": 5000,
    "New Roles Created": 10,
    "Training Hours Provided": 2000,
    "Productivity Change (%)": 12.5,
    "Employee Sentiment": "AI improved workflow but caused anxiety about job security"
}
