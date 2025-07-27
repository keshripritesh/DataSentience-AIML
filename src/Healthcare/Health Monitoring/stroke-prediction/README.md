🧠 Stroke Prediction Using Machine Learning
Predict the likelihood of a stroke based on health metrics such as age, BMI, glucose level, smoking habits, and medical history using a Random Forest classifier and SMOTE to handle class imbalance.


[!ui screenshot](assets/Screenshot_3-7-2025_144231_localhost.jpeg)
[!ui screenshot](assets/image.png)
stroke-prediction/
├── data/
│   └── stroke_data.csv           # Dataset (CSV format)
├── models/
│   └── model.pkl                 # Trained ML model
├── preprocess.py                 # Preprocessing logic (cleaning + encoding)
├── train.py                      # Training script with SMOTE + evaluation
├── predict.py                    # Prediction script for sample inputs
├── requirements.txt              # Required Python libraries
└── README.md                     # Project documentation


📊 Dataset
Source: Kaggle - Stroke Prediction Dataset

Features Include:

Age, Gender, BMI, Glucose Level

Hypertension, Heart Disease

Marital Status, Work Type

Smoking Status, Residence Type

Target: stroke (0 = No stroke, 1 = Stroke)

⚙️ Model Details
Model: RandomForestClassifier (scikit-learn)

Handling Imbalance: SMOTE (Synthetic Minority Oversampling Technique)

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix