Stock Impact Regressor by AI

This project predicts the impact of AI-related events on a company's stock performance using a regression model trained on financial and AI adoption data.
[!ui](assets/image.png)
🔍 Project Overview

The Stock Impact Regressor by AI leverages machine learning to assess how factors like:

Company R&D Spending (USD Mn)

AI Revenue (USD Mn)

AI Revenue Growth (%)

Events (e.g., product launches, acquisitions, AI breakthroughs)

affect a company’s stock movement and overall market impact.

The model uses a scikit-learn pipeline with preprocessing (scaling, encoding) and regression techniques to generate predictions.

📂 Files in this Project

preprocess.py → Handles preprocessing, feature engineering, and transformations.

train.py → Trains the regression model on the dataset and saves it under model/.

predict.py → Loads the saved model and predicts stock impact given company and event details.

model/ → Stores trained regression models.

data/ → Training dataset for AI stock impact analysis.