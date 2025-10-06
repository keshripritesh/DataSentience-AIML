"""
Utility functions for Agricultural Solutions Dashboard
"""
import streamlit as st
import requests
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config import OPENWEATHER_API_KEY, DEFAULT_LOCATIONS, CROP_PRICES, SOIL_HEALTH_RANGES

def get_weather_data(location="Delhi", demo_mode=True):
    """
    Get weather data for a location
    In demo mode, returns simulated data
    """
    if demo_mode or OPENWEATHER_API_KEY == "demo_mode":
        # Return simulated weather data
        return {
            "location": location,
            "temperature": round(random.uniform(15, 35), 1),
            "humidity": round(random.uniform(30, 90), 1),
            "description": random.choice(["Clear sky", "Few clouds", "Scattered clouds", "Light rain"]),
            "wind_speed": round(random.uniform(2, 15), 1),
            "pressure": round(random.uniform(1000, 1020), 1),
            "feels_like": round(random.uniform(15, 38), 1),
            "uv_index": random.randint(1, 10)
        }
    
    try:
        # Real API call (when API key is provided)
        coords = DEFAULT_LOCATIONS.get(location, DEFAULT_LOCATIONS["Delhi"])
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        return {
            "location": location,
            "temperature": round(data["main"]["temp"], 1),
            "humidity": round(data["main"]["humidity"], 1),
            "description": data["weather"][0]["description"].title(),
            "wind_speed": round(data["wind"]["speed"], 1),
            "pressure": round(data["main"]["pressure"], 1),
            "feels_like": round(data["main"]["feels_like"], 1),
            "uv_index": random.randint(1, 10)  # UV index requires separate API call
        }
    except Exception as e:
        st.error(f"Weather API error: {e}")
        return get_weather_data(location, demo_mode=True)

def analyze_soil_health(n, p, k, ph):
    """
    Analyze soil health based on NPK and pH values
    """
    analysis = {
        "overall_score": 0,
        "recommendations": [],
        "nutrient_status": {}
    }
    
    # Analyze Nitrogen
    if n < SOIL_HEALTH_RANGES["nitrogen"]["low"][1]:
        analysis["nutrient_status"]["nitrogen"] = {"status": "Low", "color": "🔴", "score": 2}
        analysis["recommendations"].append("Consider nitrogen-rich fertilizers or organic compost")
    elif n <= SOIL_HEALTH_RANGES["nitrogen"]["medium"][1]:
        analysis["nutrient_status"]["nitrogen"] = {"status": "Medium", "color": "🟡", "score": 6}
        analysis["recommendations"].append("Nitrogen levels are adequate")
    else:
        analysis["nutrient_status"]["nitrogen"] = {"status": "High", "color": "🟢", "score": 9}
        analysis["recommendations"].append("Excellent nitrogen levels")
    
    # Analyze Phosphorus
    if p < SOIL_HEALTH_RANGES["phosphorus"]["low"][1]:
        analysis["nutrient_status"]["phosphorus"] = {"status": "Low", "color": "🔴", "score": 2}
        analysis["recommendations"].append("Add phosphorus-rich fertilizers")
    elif p <= SOIL_HEALTH_RANGES["phosphorus"]["medium"][1]:
        analysis["nutrient_status"]["phosphorus"] = {"status": "Medium", "color": "🟡", "score": 6}
        analysis["recommendations"].append("Phosphorus levels are adequate")
    else:
        analysis["nutrient_status"]["phosphorus"] = {"status": "High", "color": "🟢", "score": 9}
        analysis["recommendations"].append("Excellent phosphorus levels")
    
    # Analyze Potassium
    if k < SOIL_HEALTH_RANGES["potassium"]["low"][1]:
        analysis["nutrient_status"]["potassium"] = {"status": "Low", "color": "🔴", "score": 2}
        analysis["recommendations"].append("Increase potassium through fertilizers or ash")
    elif k <= SOIL_HEALTH_RANGES["potassium"]["medium"][1]:
        analysis["nutrient_status"]["potassium"] = {"status": "Medium", "color": "🟡", "score": 6}
        analysis["recommendations"].append("Potassium levels are adequate")
    else:
        analysis["nutrient_status"]["potassium"] = {"status": "High", "color": "🟢", "score": 9}
        analysis["recommendations"].append("Excellent potassium levels")
    
    # Analyze pH
    if ph < SOIL_HEALTH_RANGES["ph"]["acidic"][1]:
        analysis["nutrient_status"]["ph"] = {"status": "Acidic", "color": "🔴", "score": 4}
        analysis["recommendations"].append("Soil is acidic - consider lime application")
    elif ph <= SOIL_HEALTH_RANGES["ph"]["neutral"][1]:
        analysis["nutrient_status"]["ph"] = {"status": "Neutral", "color": "🟢", "score": 10}
        analysis["recommendations"].append("Excellent pH balance")
    else:
        analysis["nutrient_status"]["ph"] = {"status": "Alkaline", "color": "🟡", "score": 6}
        analysis["recommendations"].append("Soil is alkaline - consider sulfur application")
    
    # Calculate overall score
    total_score = sum([analysis["nutrient_status"][nutrient]["score"] for nutrient in analysis["nutrient_status"]])
    analysis["overall_score"] = round(total_score / 4, 1)
    
    return analysis

def get_crop_price_data(crop_name):
    """
    Get current market price data for a crop
    """
    crop_lower = crop_name.lower()
    if crop_lower in CROP_PRICES:
        price_data = CROP_PRICES[crop_lower].copy()
        
        # Add some realistic variation
        variation = random.uniform(-0.02, 0.02)  # ±2% variation
        price_data["current"] = round(price_data["current"] * (1 + variation))
        
        return price_data
    else:
        # Default price data for unknown crops
        return {
            "current": random.randint(2000, 8000),
            "trend": random.choice(["up", "down", "stable"]),
            "change": round(random.uniform(-5, 10), 1)
        }

def generate_crop_price_forecast(crop_name, days=30):
    """
    Generate price forecast for a crop (simulated)
    """
    base_price = get_crop_price_data(crop_name)["current"]
    dates = [datetime.now() + timedelta(days=i) for i in range(days)]
    
    prices = []
    current_price = base_price
    
    for _ in range(days):
        # Add some realistic price movement
        change = random.uniform(-0.05, 0.05)  # ±5% daily change
        current_price *= (1 + change)
        prices.append(round(current_price, 2))
    
    return pd.DataFrame({
        "Date": dates,
        "Price": prices
    })

def translate_text(text, target_lang="hi", demo_mode=True):
    """
    Translate text to target language
    In demo mode, returns basic translations for key terms
    """
    if demo_mode or target_lang == "en":
        return text
    
    # Basic translation dictionary for demo
    translations = {
        "hi": {  # Hindi
            "Crop Recommendation": "फसल सिफारिश",
            "Dashboard": "डैशबोर्ड",
            "Weather": "मौसम",
            "Soil Health": "मिट्टी स्वास्थ्य",
            "Price Forecast": "मूल्य पूर्वानुमान",
            "Temperature": "तापमान",
            "Humidity": "आर्द्रता",
            "Rainfall": "वर्षा",
            "Nitrogen": "नाइट्रोजन",
            "Phosphorus": "फास्फोरस",
            "Potassium": "पोटैशियम",
            "Recommend Crop": "फसल सुझाएं"
        },
        "te": {  # Telugu
            "Crop Recommendation": "పంట సిఫార్సు",
            "Dashboard": "డాష్‌బోర్డ్",
            "Weather": "వాతావరణం",
            "Soil Health": "మట్టి ఆరోగ్యం"
        }
    }
    
    if target_lang in translations and text in translations[target_lang]:
        return translations[target_lang][text]
    
    return text  # Return original if no translation available

def get_seasonal_recommendations(month=None):
    """
    Get seasonal crop recommendations based on current month
    """
    if month is None:
        month = datetime.now().month
    
    seasons = {
        "winter": {
            "months": [12, 1, 2],
            "crops": ["wheat", "mustard", "gram", "pea", "lentil"],
            "description": "Winter season - ideal for Rabi crops"
        },
        "summer": {
            "months": [3, 4, 5],
            "crops": ["watermelon", "muskmelon", "cucumber", "fodder"],
            "description": "Summer season - focus on heat-resistant crops"
        },
        "monsoon": {
            "months": [6, 7, 8, 9],
            "crops": ["rice", "maize", "cotton", "sugarcane", "jute"],
            "description": "Monsoon season - ideal for Kharif crops"
        },
        "post_monsoon": {
            "months": [10, 11],
            "crops": ["chickpea", "mustard", "barley", "wheat"],
            "description": "Post-monsoon - prepare for winter crops"
        }
    }
    
    for season, data in seasons.items():
        if month in data["months"]:
            return data
    
    return seasons["winter"]  # Default fallback
