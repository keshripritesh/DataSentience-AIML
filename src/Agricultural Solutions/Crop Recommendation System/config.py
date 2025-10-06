# Configuration file for Agricultural Solutions Dashboard
# Using demo mode - replace with real API keys for production

# Weather API Configuration (Demo mode - using mock data)
OPENWEATHER_API_KEY = "demo_mode"  # Replace with real API key from https://openweathermap.org/api
WEATHER_API_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Default locations for weather data
DEFAULT_LOCATIONS = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714}
}

# Crop price data (simulated - in real app, connect to market API)
CROP_PRICES = {
    "rice": {"current": 2500, "trend": "up", "change": 5.2},
    "maize": {"current": 1800, "trend": "down", "change": -2.1},
    "chickpea": {"current": 5500, "trend": "up", "change": 8.3},
    "kidneybeans": {"current": 6200, "trend": "stable", "change": 0.5},
    "pigeonpeas": {"current": 4800, "trend": "up", "change": 3.7},
    "mothbeans": {"current": 3200, "trend": "down", "change": -1.8},
    "mungbean": {"current": 7500, "trend": "up", "change": 12.4},
    "blackgram": {"current": 6800, "trend": "stable", "change": 1.2},
    "lentil": {"current": 5200, "trend": "up", "change": 4.6},
    "pomegranate": {"current": 8500, "trend": "up", "change": 6.8},
    "banana": {"current": 2200, "trend": "stable", "change": -0.3},
    "mango": {"current": 4500, "trend": "down", "change": -3.2},
    "grapes": {"current": 3800, "trend": "up", "change": 2.9},
    "watermelon": {"current": 1200, "trend": "stable", "change": 0.8},
    "muskmelon": {"current": 1800, "trend": "up", "change": 1.5},
    "apple": {"current": 6500, "trend": "up", "change": 4.2},
    "orange": {"current": 3200, "trend": "stable", "change": 0.9},
    "papaya": {"current": 2800, "trend": "down", "change": -2.5},
    "coconut": {"current": 2500, "trend": "up", "change": 3.1},
    "cotton": {"current": 4200, "trend": "up", "change": 7.8},
    "jute": {"current": 3500, "trend": "stable", "change": 1.1},
    "coffee": {"current": 15500, "trend": "up", "change": 9.5}
}

# Language support
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "हिंदी (Hindi)",
    "te": "తెలుగు (Telugu)",
    "ta": "தமிழ் (Tamil)",
    "kn": "ಕನ್ನಡ (Kannada)",
    "bn": "বাংলা (Bengali)",
    "gu": "ગુજરાતી (Gujarati)",
    "mr": "मराठी (Marathi)",
    "pa": "ਪੰਜਾਬੀ (Punjabi)"
}

# Soil health parameters
SOIL_HEALTH_RANGES = {
    "nitrogen": {"low": (0, 40), "medium": (40, 80), "high": (80, 140)},
    "phosphorus": {"low": (5, 25), "medium": (25, 60), "high": (60, 145)},
    "potassium": {"low": (5, 50), "medium": (50, 100), "high": (100, 205)},
    "ph": {"acidic": (3.0, 6.0), "neutral": (6.0, 7.5), "alkaline": (7.5, 10.0)}
}
