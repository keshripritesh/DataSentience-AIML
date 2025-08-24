# 🚢 Titanic Survival Prediction

This project performs data exploration, cleaning, and visualization on the famous Titanic dataset. The aim is to understand survival patterns based on features like gender, age, passenger class, fare, and embarkation port.

---
# 📌 Objective

To analyze the Titanic dataset and discover factors that influenced passenger survival. The project focuses on:
- Handling missing data
- Visualizing survival distributions
- Understanding correlations between features

📂 Dataset

Source: Titanic-Dataset

Shape: (891, 12)

**Target Variable:**
- Survived → (0 = Did Not Survive, 1 = Survived)

**Features:**
- PassengerId
- Pclass (Passenger class: 1, 2, 3)
- Name
- Sex
- Age
- SibSp (Siblings/Spouses aboard)
- Parch (Parents/Children aboard)
- Ticket
- Fare
- Cabin
- Embarked (Port of embarkation: C, Q, S)

---

# 🔍 Exploratory Data Analysis (EDA)

- Checked dataset structure with .info() and .describe()
- Counted missing values → Age, Cabin, Embarked
- Removed duplicates (none found)
-Examined correlations with survival (Pclass, Fare, Age, etc.)

Visualizations created using Seaborn & Matplotlib:
- Correlation heatmaps
- Survival counts (overall, by gender, class, embarkation)
- Age distributions (with and without missing values filled)
- Fare distribution (boxplots with log scale)
- Pair plots for multi-variable patterns

--- 
# ⚙️ Data Cleaning
- Age → filled with median age
- Embarked → filled with mode
- Cabin → replaced NaN with “NoCabin”
- Final dataset: 0 missing values ✅

---

# 📈 Key Insights

- Gender: Females had higher survival rates.
- Pclass: 1st class passengers were more likely to survive.
- Fare: Higher ticket prices correlated with higher survival chances.
- Embarked: Survival varied slightly across ports.
- Age: Younger passengers had slightly better survival rates.

---

🚀 How to Run
```
Upload the dataset (Titanic-Dataset.csv) to your Colab environment.

Run the notebook cells in sequence.

Modify visualization parameters (palette, cmap, etc.) for custom plots.
```
---

# 👤 Author

GitHub: [archangel2006](https://github.com/archangel2006)

Feel free to ⭐ this repository or fork it to extend the analysis with feature engineering and machine learning models! 🚀
