# 🛡️ CrimeLens – Crime Report Classifier

**CrimeLens** is a machine learning project that classifies news headlines as **CRIME** or **NO_CRIME**.  
It uses **TF-IDF vectorization** and **Logistic Regression** to detect crime-related headlines from textual input.
[!ui](assets/image.png)
---

## 📂 Project Structure
project/
│
├── data/
│ └── CrimeVsNoCrimeArticles.csv # Dataset
│
├── model/
│ └── crime_model.joblib # Trained model (after running train.py)
│
├── preprocess.py # Data cleaning & preprocessing
├── train.py # Model training script
├── predict.py # Prediction script
└── README.md # Documentation


---

## ⚙️ Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/crime-report-classifier.git
cd crime-report-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



python predict.py --text "Armed robbery reported in downtown area"

CRIME   Armed robbery reported in downtown area
