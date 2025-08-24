# 🩸 Blood Donation Volume Prediction

This project predicts the **volume of blood (in pints)** a donor is likely to donate in the future, based on their donation history and demographics.  
It is the second model built on the same dataset and complements the **Donor Availability Prediction** model.  
[!ui](assets/image.png)
---

## 📂 Project Structure
data/
└── blood_donor_dataset.csv
model/
└── donation_volume.pkl
preprocess.py
train.py
predict.py

---

## 📊 Dataset
The dataset contains **10,000 donor records**.  
Key features used in this project:  
- `months_since_first_donation` → Time since first donation (months)  
- `number_of_donation` → Total donations made  
- `city` → Donor's city  
- `blood_group` → Donor’s blood type  
- **Target**: `pints_donated` → Total pints of blood donated  

---

## ⚙️ Preprocessing (`preprocess.py`)
- Selects relevant features and target.  
- Encodes categorical columns (`city`, `blood_group`).  
- Splits dataset into training/testing sets.  

Run:
```bash
python preprocess.py

🤖 Training (train.py)

Uses Random Forest Regressor.

Evaluates with RMSE (Root Mean Squared Error) and R² score.

Saves trained model and encoders to model/donation_volume.pkl.

