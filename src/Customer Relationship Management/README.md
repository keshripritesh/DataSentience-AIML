# Customer Churn Prediction 

This project focuses on building a machine learning model to predict customer churn using structured client and pricing data. The goal is to assist businesses in identifying customers who are likely to discontinue their services and enable them to take preventive actions based on data-driven insights.

## Table of Contents

- [Overview](#overview)
- [Approach and Methodology](#approach-and-methodology)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Future Scope](#future-scope)

## Overview

Customer churn refers to when a customer stops doing business with a company. Retaining existing customers is more cost-effective than acquiring new ones. This project applies machine learning techniques to analyze customer behavior and predict churn using historical datasets. It includes data preprocessing, model building, evaluation, and generation of predictions on new data.

## Approach and Methodology

1. **Exploratory Data Analysis (EDA)**  
   - Understand data distribution, missing values, correlations, and key features.

2. **Data Preprocessing**  
   - Encoding categorical variables, feature scaling, handling null values, and merging datasets.

3. **Model Building**  
   - Algorithms evaluated include Logistic Regression, Random Forest, and XGBoost.  
   - Models were selected based on metrics such as accuracy, precision, recall, and F1 score.

4. **Prediction Generation**  
   - Predictions were generated on unseen data from `prediction_data.csv` and stored in `final_outputdata.csv`.

5. **Documentation**  
   - The process and findings are summarized in the `final report.pdf`.

## Project Structure
```
  Customer-Churn_Prediction/
  ├── Notebook/
  │   └── Customer churn prediction.ipynb 
  ├── datsets/
  │   ├── .ipynb_checkpoints/
  │   ├── cleaned_data.csv
  │   ├── client_data.csv
  │   ├── prediction_data.csv
  │   └── price_data.csv
  ├── .gitignore
  ├── Contributing.md
  ├── LICENSE
  ├── README.md
  ├── final report.pdf     
  ├── final_outputdata.csv  
  └── requirements.txt
```

## Datasets

- `cleaned_data.csv`: Preprocessed data for training/testing  
- `client_data.csv`: Raw client attributes and usage information  
- `prediction_data.csv`: Data for which churn needs to be predicted  
- `price_data.csv`: Package/pricing-related data  

## Installation

To set up this project locally:

1. **Clone the repository:**
   ```
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```
   pip install -r requirements.txt
   ```

4. **Launch the Jupyter notebook:**
   ```
   jupyter notebook Customer churn prediction.ipynb
   ```

5. **Review the report and results:**
   Open `final report.pdf` and `final_outputdata.csv` for insights and output.

## Dependencies

The project requires the Python packages (mentioned in  `requirements.txt`)

## Contributing

Contributions are welcome. Please read the `CONTRIBUTING.md` file for guidelines.

## License

This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

## Future Scope

- Incorporate real-time prediction using APIs or dashboards.  
- Add customer segmentation based on behavior for tailored marketing.  
- Integrate time-series data or customer feedback for deeper analysis.
  
###### Thank you for checking out the Customer Churn Prediction project! Feel free to explore and contribute.
