# 🛒 Cart Abandonment Predictor & Email Simulator

This project predicts **cart abandonment behavior** using real-world e-commerce session data and simulates basic **email recovery strategies** to help businesses understand and reduce abandonment rates.

## Project Overview

- Predicts whether a user will **abandon their cart** based on session behavior.
- Uses machine learning models trained on enhanced features.
- Includes a basic **email simulator** to model how timely emails can help recover sales.
- Built with **Streamlit** for real-time input and visualization.

---

## Dataset Source

The dataset was obtained from Medium blog.

Features in dataset:
<pre>
ID: The session id of the customer.

Is_Product_Details_viewed: Whether the customer is viewing the product details or not.

Session_Activity_Count: How many times a customer is going to the different pages.

No_Items_Added_InCart: Number of items in cart.

No_Items_Removed_FromCart: Number of items removed from the cart.

No_Cart_Viewed: How many times the customer is going to the cart page.

No_Checkout_Confirmed: How many times the checkout has been confirmed successfully by the customer.

No_Checkout_Initiated: How many times the checkout(successful as well as unsuccess) is being done by the user.

No_Cart_Items_Viewed: How many times a user is viewing the product from cart.

No_Customer_Login: Number of times the customer had did log in.

No_Page_Viewed: Number of pages viewed by the customer.

Customer_Segment_Type: The customer falls under which category,i.e, 0 for Target Customer, 1 for Loyal Customer, and 2 for Untargeted customer.

Cart_Abandoned: Whether the customer is doing cart abandonment or not. This is the target variable that we need to predict.</pre>   

---

## ML Models Used

Trained 3 models and compared performance:

<pre>

Logistic Regression     : 98.13%

Random Forest           : 97.55%

XG Boost                : 98.13%
</pre>
---

## Email Simulator

The email simulator is a lightweight tool to test recovery actions:

- Inputs are passed to the trained model to detect abandonment.
- If predicted as “abandoned”, the system triggers a mock email alert.
- Designed to showcase basic **user re-engagement logic**.

> You can enhance it by integrating real-time campaign tools or analytics dashboards.

---

## Tech Stack

- Python (Numpy,Pandas, Scikit-learn, Streamlit)
- Jupyter Notebook (Model development)
- Pickle (Model storage)
- Matplotlib / Seaborn (EDA)

---

## Folder Structure
<pre> 
cart_abandonment/
│
├── data/
│ └── data_cart_abandonment.csv   #source data
├── model/
│ └── model.pkl                   #trained models
├── notebook/
│ └── model.ipynb                 #Notebook with EDA + training
├── simulation.py                 #email and streamlit logic
├── requirements.txt              #Python dependencies
└── README.md </pre>

---

> GitHub does not render Jupyter notebook outputs. To view the notebook properly, use nbviewer.

---

## Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the simulator:
    ```bash
   streamlit run simulation.py
---

## Contact
For queries or collaboration: shreya.ramesh22@gmail.com


