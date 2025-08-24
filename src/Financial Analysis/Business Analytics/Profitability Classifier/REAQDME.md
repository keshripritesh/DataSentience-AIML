📊 Profitability Classifier
🔍 Project Overview

This project predicts whether a company is “High Profit” or “Low Profit” based on its financial and categorical attributes such as Sales, Assets, Market Value, Industry, and Headquarters.

The dataset used is the Forbes Global 2000 Companies (2025), which contains 2000 companies ranked by revenue, profit, assets, and market value.
[!ui](assets/image.png)
This project follows the SSOC format:

data/ → contains the dataset (Forbes_2000_Companies_2025.csv)

model/ → stores the trained model (profit_classifier.pkl)

preprocess.py → data cleaning, encoding, scaling, and train-test split

train.py → trains the model and evaluates performance

predict.py → loads the trained model and makes predictions


Profitability Classifier/
│── data/
│   └── Forbes_2000_Companies_2025.csv
│── model/
│   └── profit_classifier.pkl
│── preprocess.py
│── train.py
│── predict.py
│── README.md


pandas
scikit-learn
joblib


pyhton train.py
python predict.py
