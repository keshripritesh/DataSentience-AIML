# 🎓 Student Habits & Exam Performance 

This project performs data exploration, cleaning, and visualization on a dataset of student habits and academic performance. The aim is to understand how lifestyle factors like study hours, sleep, social media, and exercise relate to exam scores.

---

# 📌 Objective

To analyze the student habits dataset and explore factors influencing exam performance. The project focuses on:
- Handling missing data
- Exploring dataset structure and feature distributions
- Understanding correlations between habits and exam scores
- Visualizing trends using Seaborn & Matplotlib

---

# 📂 Dataset

Source: student_habits_performance.csv

Shape: (1000, 16)

**Target Variable**:

- exam_score → (0–100)

**Features**:

- student_id
- age
- gender
- study_hours_per_day
- social_media_hours
- netflix_hours
- part_time_job
- attendance_percentage
- sleep_hours
- diet_quality
- exercise_frequency
- internet_quality
- mental_health_rating
- extracurricular_participation
- parental_education_level (some missing values)

---

# 🔍 Exploratory Data Analysis (EDA)
- Checked dataset structure with .info(), .head(), .describe()
- Counted missing values → primarily in parental_education_level
- Verified no duplicate entries
- Examined correlations between habits and exam_score

Visualizations created with Seaborn & Matplotlib:
- Correlation heatmaps
- Distribution plots of exam scores
- Scatterplots (study hours, social media, Netflix, sleep vs. exam score)
- Boxplots for categorical habits (diet, internet quality, exercise, etc.)

---

# ⚙️ Data Cleaning

- Handled missing values in parental_education_level (tested imputation vs. dropping)
- Created a missingness indicator for analysis
- Standardized categorical features for consistency
- Final dataset prepared with cleaned features and no duplicates ✅

---

# 📈 Key Analysis Steps

- Explore numerical vs categorical variables separately
- Compare distributions across habits
- Identify strongest correlations with exam performance

---

# 🚀 How to Run

```
Upload student_habits_performance.csv to your Colab environment.

Run the notebook cells in sequence

Modify visualization parameters (palette, cmap, etc.) to customize plots.
```
---

# 👤 Author

GitHub: [archangel2006](https://github.com/archangel2006)

