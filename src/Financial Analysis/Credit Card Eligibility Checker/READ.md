# 💳 Credit Card Eligibility Checker

A professional, interactive web application built with Streamlit that helps users determine their credit card eligibility and provides personalized recommendations.

## ✨ Features

- **🎯 Smart Eligibility Assessment**: Advanced algorithm considering multiple factors
- **📊 Credit Score Estimation**: Real-time credit score calculation with visual gauge
- **💳 Personalized Card Recommendations**: Tier-based card suggestions (Basic, Silver, Gold, Platinum)
- **📈 Financial Analysis Dashboard**: Comprehensive financial metrics and insights
- **💡 Improvement Tips**: Personalized suggestions to enhance eligibility
- **🎨 Modern UI/UX**: Professional, responsive design with interactive elements
- **📱 Mobile Responsive**: Optimized for all device sizes

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd "src/Financial Analysis/Credit Card Eligibility Checker"
   ```

2. **Create and activate a virtual environment (recommended):**
   
   **Create Virtual Environment:**
   ```bash
   python3 -m venv venv
   ```
   
   **Activate the virtual environment:**
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Navigate to the Scripts directory:**
   ```bash
   cd Scripts
   ```

5. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Access the application:**
   Open your browser and go to `http://localhost:8501`

## 📊 How It Works

### Eligibility Factors

The application evaluates eligibility based on:

- **Age**: 18-65 years (optimal: 25-45)
- **Income**: Minimum ₹15,000/month
- **Employment**: Salaried or Self-employed
- **Experience**: Work experience in years
- **Existing Cards**: Current credit card portfolio
- **Location**: City tier classification

### Credit Score Calculation

Our proprietary algorithm estimates credit scores (300-850) based on:

- **Age Profile** (20% weightage)
- **Income Level** (30% weightage)
- **Employment Stability** (20% weightage)
- **Work Experience** (15% weightage)
- **Credit Portfolio** (10% weightage)
- **Geographic Location** (5% weightage)

### Card Tiers

| Tier | Income Requirement | Features |
|------|-------------------|----------|
| **Platinum** | ₹2,00,000+ | Premium benefits, high limits, luxury perks |
| **Gold** | ₹1,00,000+ | Enhanced rewards, good benefits |
| **Silver** | ₹50,000+ | Standard benefits, cashback options |
| **Basic** | ₹20,000+ | Entry-level features, building credit |

## 🎯 Application Sections

### 1. Personal Information Form
- User-friendly input fields with validation
- Real-time feedback and helpful tooltips
- Smart defaults for quick testing

### 2. Credit Score Dashboard
- Interactive gauge visualization
- Score interpretation and ranges
- Real-time score updates

### 3. Eligibility Results
- Comprehensive eligibility assessment
- Detailed explanation of decision factors
- Improvement suggestions for rejected applications

### 4. Card Recommendations
- Personalized card suggestions based on profile
- Detailed card information (fees, limits, rewards)
- Approval probability estimates

### 5. Financial Analysis
- Key financial metrics visualization
- Debt-to-income ratio analysis
- Credit utilization insights

### 6. Improvement Tips
- Personalized financial advice
- Credit score improvement strategies
- Best practices for credit management

## 🛠️ Technical Architecture

### File Structure
```
Credit Card Eligibility Checker/
├── README.md
├── requirements.txt
└── Scripts/
    ├── streamlit_app.py      # Main application interface
    └── eligibility_logic.py  # Business logic and calculations
```

### Technologies Used

- **Frontend**: Streamlit with custom CSS styling
- **Visualizations**: Plotly for interactive charts and gauges
- **Data Processing**: Pandas and NumPy for calculations
- **Styling**: Custom CSS with gradient designs and animations

## 🎨 UI/UX Features

- **Professional Design**: Modern gradient backgrounds and card layouts
- **Interactive Elements**: Hover effects and smooth transitions
- **Responsive Layout**: Adapts to different screen sizes
- **Visual Feedback**: Color-coded results and progress indicators
- **Accessibility**: Clear typography and intuitive navigation

## 📈 Business Logic

### Eligibility Algorithm
```python
def check_eligibility(age, income, employment_status, experience, existing_cards, city_tier):
    # Multi-factor assessment
    # Returns: (eligible, message, tier)
```

### Credit Score Estimation
```python
def calculate_credit_score(age, income, employment_status, experience, existing_cards, city_tier):
    # Proprietary scoring algorithm
    # Returns: estimated_score (300-850)
```

## 🔐 Privacy & Security

- **Local Processing**: All calculations performed locally
- **No Data Storage**: User information is not stored or transmitted
- **Privacy First**: Complete data privacy and security

## 🚀 Future Enhancements

- [ ] Integration with actual credit bureaus
- [ ] Advanced machine learning models
- [ ] Comparison with real bank offers
- [ ] Credit score tracking over time
- [ ] Email/SMS notifications for offers
- [ ] Multi-language support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of the DataSentience-AIML repository and follows the same licensing terms.

## 🏆 Acknowledgments

- Built as part of the DataSentience-AIML project
- Inspired by real-world financial assessment tools
- Designed for educational and demonstration purposes

## 📧 Support

For questions or issues, please open an issue in the main repository or contact the maintainers.

---

**⚠️ Disclaimer**: This tool provides estimated eligibility based on basic criteria. Actual credit card approval depends on comprehensive credit checks, bank policies, and additional factors not considered in this simplified model. Always consult with financial institutions for accurate eligibility assessment.