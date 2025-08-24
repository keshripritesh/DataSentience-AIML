Company Valuation Classifier

This project is designed to classify companies as Overvalued or Undervalued based on their financial fundamentals. The dataset is derived from the Forbes Global 2000 Companies 2025, which contains global leaders across industries with detailed financial attributes.
[!ui](assets/image.png)
🔍 Problem Statement

The main objective is to predict whether a company is Overvalued or Undervalued by analyzing:

Sales ($B)

Assets ($B)

Profits ($B)

Industry

Market Value ($B)

This is a binary classification problem, where the target label is company valuation status.

🏗️ Approach
Data Preprocessing (preprocess.py)

Load the dataset and handle missing values.

Encode categorical features such as Industry.

Define the target column:

If Market Value / Sales > threshold → Overvalued

Else → Undervalued
(Here the threshold is chosen as the median of the Market Value/Sales ratio).

Split the dataset into training and testing sets.

Model Training (train.py)

Train a Logistic Regression Classifier (chosen for interpretability and ease of use in financial analysis).

Save the trained model to disk for predictions.

Evaluate performance with accuracy, confusion matrix, and classification report.

Prediction (predict.py)

Load the trained model.

Accept input (sales, assets, profits, industry, and market value).

Output prediction: Overvalued or Undervalued.

📊 Example Usage