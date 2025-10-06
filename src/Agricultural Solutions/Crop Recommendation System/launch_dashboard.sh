#!/bin/bash
# Agricultural Dashboard Launcher
# This script ensures the app runs with the correct conda environment

echo "🌾 Starting Agricultural Solutions Dashboard..."
echo "📦 Activating agri-dashboard environment..."

cd "/Users/siya/Desktop/DataSentience-AIML/src/Agricultural Solutions/Crop Recommendation System"

# Run with the correct conda environment
conda run -n agri-dashboard streamlit run app.py --server.headless false
