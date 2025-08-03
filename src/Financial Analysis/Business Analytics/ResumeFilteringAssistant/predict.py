import pandas as pd
import pickle
from preprocess import clean_and_encode

def predict_from_dataframe(df):
    # Load model, scaler, and training columns
    with open('models/candidate_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('models/train_columns.pkl', 'rb') as f:
        train_columns = pickle.load(f)

    # Clean and encode input
    df_clean = clean_and_encode(df)

    # Handle missing columns efficiently
    missing_cols = [col for col in train_columns if col not in df_clean.columns]
    extra_cols = [col for col in df_clean.columns if col not in train_columns]

    # Add missing columns as 0
    df_missing = pd.DataFrame(0, index=df_clean.index, columns=missing_cols)
    df_clean = pd.concat([df_clean, df_missing], axis=1)

    # Drop extra columns
    df_clean = df_clean.drop(columns=extra_cols)

    # Reorder columns to match training
    df_clean = df_clean[train_columns]

    # Scale and predict
    X_scaled = scaler.transform(df_clean)
    preds = model.predict(X_scaled)

    df['Prediction'] = preds
    return df[['Name', 'Prediction']] if 'Name' in df.columns else df[['Prediction']]

if __name__ == '__main__':
    # Minimal working example (tweak based on actual columns)
    sample_data = pd.DataFrame([{
        'Name': 'Shahan',
        'Age': 24,
        'Gender': 'Male',
        'Type of Graduation/Post Graduation': 'B.Tech',
        'Marital status': 'Single',
        'Mode of interview given by candidate?': 'Offline',
        'Pre Interview Check': 'Done',
        'Fluency in English based on introduction': 'Good',
        'Confidence based on Introduction (English)': 'Average',
        'Confidence based on the topic given  ': 'Good',
        'Confidence Based on the PPT Question': 'Average',
        'Confidence based on the sales scenario': 'Poor',
        'Structured Thinking (In regional only)': 'Average',
        'Structured Thinking Based on the PPT Question': 'Good',
        'Structured Thinking( Call pitch)': 'Good',
        'Regional fluency based on the topic given  ': 'Good',
        'Regional fluency Based on the PPT Question': 'Good',
        'Regional fluency based on the  sales scenario': 'Average',
        'Does the candidate has mother tongue influence while speaking english.': 'No',
        'Has acquaintance in Company and has spoken to him/her before applying?': 'Yes',
        'Candidate Status': 'Shortlisted',
        'Last Fixed CTC (lakhs)': 4.5,
        'Currently Employed': 'Yes',
        'Experienced candidate - (Experience in months)': 24,
        'Experienced Candidate (Nature of work)': 'Sales',
        'What was the type of Role?': 'Inside Sales',
        'How many slides candidate have submitted in PPT?': 5,
        'Call-pitch Elements used during the call Sales Scenario': 'Good',
        'But, my child\'s exam are going on now, so we will keep the counselling session after the exams get over.(Time: Favourable pitch: Counsellor hype)': 'Handled',
        'Let me discuss it with my child': 'Handled',
        'Sir being in education industry I know this is a marketing gimmick and at the end of the day you\'ll be selling the app.': 'Not Handled',
        'Role acceptance': 'Yes',
        'Interview Verdict': 'Selected',
        'Candidate is willing to relocate': 'Yes',
        'Role Location to be given to the candidate': 'Delhi'
    }])

    result = predict_from_dataframe(sample_data)
    print(result)
