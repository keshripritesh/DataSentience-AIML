"""
Demo script to test the Startup Success Predictor functionality
"""

import pandas as pd
import joblib
import os

def test_model_functionality():
    """Test the trained model with sample data"""
    
    print("🚀 Testing Startup Success Predictor...")
    
    # Check if model files exist
    model_path = 'models/rf_model.pkl'
    features_path = 'models/feature_columns.pkl'
    
    if not os.path.exists(model_path):
        print("❌ Model not found. Training model first...")
        os.system('python train_model.py')
    
    # Load model and features
    print("📦 Loading model and features...")
    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Number of features: {len(feature_columns)}")
    print(f"🎯 Model type: {type(model).__name__}")
    
    # Create sample startup data
    print("\n🏢 Testing with sample startup data...")
    
    sample_startups = [
        {
            'name': 'TechNova AI',
            'funding_total_usd': 5000000,
            'funding_rounds': 3,
            'location': 'California',
            'industry': 'Software'
        },
        {
            'name': 'GreenStart Solutions', 
            'funding_total_usd': 500000,
            'funding_rounds': 1,
            'location': 'Texas',
            'industry': 'Consulting'
        },
        {
            'name': 'MobileFirst Corp',
            'funding_total_usd': 15000000,
            'funding_rounds': 4,
            'location': 'New York',
            'industry': 'Mobile'
        }
    ]
    
    for i, startup in enumerate(sample_startups, 1):
        print(f"\n📊 Sample {i}: {startup['name']}")
        
        # Create input data matching model features
        input_data = pd.DataFrame({
            'latitude': [37.7749],
            'longitude': [-122.4194],
            'relationships': [5],
            'funding_rounds': [startup['funding_rounds']],
            'funding_total_usd': [startup['funding_total_usd']],
            'milestones': [2],
            'is_CA': [1 if startup['location'] == 'California' else 0],
            'is_NY': [1 if startup['location'] == 'New York' else 0],
            'is_MA': [0],
            'is_TX': [1 if startup['location'] == 'Texas' else 0],
            'is_otherstate': [0],
            'age_first_funding_year': [1.5],
            'age_last_funding_year': [3.0],
            'age_first_milestone_year': [2.0],
            'age_last_milestone_year': [4.0],
            'has_VC': [1],
            'has_angel': [1],
            'has_roundA': [1],
            'has_roundB': [1 if startup['funding_rounds'] >= 2 else 0],
            'has_roundC': [1 if startup['funding_rounds'] >= 3 else 0],
            'has_roundD': [1 if startup['funding_rounds'] >= 4 else 0],
            'avg_participants': [2.5],
            'is_top500': [0],
        })
        
        # Set industry flags
        industry_map = {
            'Software': 'is_software',
            'Mobile': 'is_mobile', 
            'Consulting': 'is_consulting'
        }
        
        # Initialize all industry flags to 0
        for flag in ['is_software', 'is_web', 'is_mobile', 'is_enterprise', 
                    'is_advertising', 'is_gamesvideo', 'is_ecommerce', 
                    'is_biotech', 'is_consulting', 'is_othercategory']:
            input_data[flag] = 0
        
        # Set the relevant industry flag
        if startup['industry'] in industry_map:
            input_data[industry_map[startup['industry']]] = 1
        
        # Add missing columns with default values
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training data
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction_proba = model.predict_proba(input_data)[0]
        success_probability = prediction_proba[1]
        prediction = model.predict(input_data)[0]
        
        # Display results
        print(f"   💰 Funding: ${startup['funding_total_usd']:,}")
        print(f"   🔄 Rounds: {startup['funding_rounds']}")
        print(f"   📍 Location: {startup['location']}")
        print(f"   🏭 Industry: {startup['industry']}")
        print(f"   🎯 Success Probability: {success_probability*100:.1f}%")
        print(f"   📈 Prediction: {'✅ Success' if prediction == 1 else '❌ High Risk'}")
    
    print(f"\n🎉 Model testing completed successfully!")
    print(f"🌐 Run 'streamlit run streamlit_app.py' to launch the web interface!")

if __name__ == "__main__":
    test_model_functionality()
