# 🩸 Donor Availability Prediction

This project predicts whether a blood donor is currently **available for donation (Yes/No)** using their donation history and demographic details.  
It is built in the **SSOC style** with a clean modular structure (`preprocess.py`, `train.py`, `predict.py`) and stores the dataset in `data/` and the trained model in `model/`.

[!ui](assets/image.png)
---

## 📂 Project Structure
data/
└── blood_donor_dataset.csv
model/
└── donor_availability.pkl
preprocess.py
train.py
predict.py

---

## 📊 Dataset
The dataset contains **10,000 donor records** with the following key fields:

- `months_since_first_donation` → Number of months since the donor’s first donation  
- `number_of_donation` → Total donations made by the donor  
- `pints_donated` → Total pints of blood donated  
- `city` → City where the donor resides  
- `blood_group` → Blood group of the donor  
- `availability` → Target variable (**Yes/No**)  

---

## ⚙️ Preprocessing (`preprocess.py`)
- Selects relevant features (`months_since_first_donation`, `number_of_donation`, `pints_donated`, `city`, `blood_group`).  
- Encodes categorical columns (`city`, `blood_group`, `availability`).  
- Splits data into train/test sets.  

Run:
```bash
python preprocess.py

🤖 Training (train.py)

Uses Random Forest Classifier.

Evaluates model performance with accuracy and classification report.

Saves trained model + encoders into model/donor_availability.pkl.
🔮 Prediction (predict.py)

Predicts availability for a given donor profile.
