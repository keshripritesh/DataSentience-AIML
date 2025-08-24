🎧 Ear Discomfort Predictor
📌 Overview

The Ear Discomfort Predictor is a machine learning project that analyzes survey responses to predict whether a person is likely to experience ear discomfort after using headphones.
The dataset comes from a Hearing Well-being Survey, which includes insights into headphone usage habits, hearing awareness, age groups, and willingness to adopt hearing-care solutions.
[!ui](assets/image.png)
This project is part of SSOC (Student Summer of Code) contributions and aims to support research into safe headphone use and personalized hearing-care recommendations.

🚀 Project Goals

Predict ear discomfort likelihood (Yes, No, Maybe, Occasionally, etc.) based on survey features.

Help in understanding risk factors like:

Duration of daily headphone use.

Age group influence.

Perceived barriers to hearing tests.

Interest in hearing test apps.

Support awareness campaigns by identifying high-risk groups.

📂 Dataset

Name: Hearing Well-being Survey Report

Rows: 387

Columns: 14

🔑 Key Features Used:

Perceived_Hearing_Meaning – What hearing means to the participant.

Hearing_FOMO – Fear of missing out due to hearing issues.

Hearing_Test_Barrier – Barriers preventing hearing tests (cost, awareness, shame, etc.).

Daily_Headphone_Use – Duration of headphone use.

Belief_Early_Hearing_Care – Numeric scale indicating belief in early care.

Age_group – Age range of the participant.

Awareness_on_hearing_and_Willingness_to_invest – Awareness and readiness to invest in hearing care.

Target: Ear_Discomfort_After_Use

⚙️ Project Structure
Ear Discomfort Predictor/
│── data/
│   └── Hearing well-being Survey Report.csv
│── model/
│   └── ear_discomfort_predictor.pkl
│── preprocess.py   # Preprocessing & encoding
│── train.py        # Training the model
│── predict.py      # Running predictions
│── README.md


🧠 Model Details

Algorithm: Random Forest Classifier

Why Random Forest?

Works well with categorical + numerical features.

Handles survey data without heavy preprocessing.

Provides good interpretability.