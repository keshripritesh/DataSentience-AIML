import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from eligibility_logic import check_eligibility, calculate_credit_score, get_card_recommendations, get_financial_tips

# Page Configuration
st.set_page_config(
    page_title="💳 Credit Card Eligibility Checker",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegant pastel styling
st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #e8f5e8 0%, #f0f8ff 50%, #fff0f5 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            color: #2c3e50;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .card {
            background: linear-gradient(145deg, #fafafa 0%, #ffffff 100%);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            margin: 1.5rem 0;
            border-left: 4px solid #a8d8a8;
            border-top: 1px solid rgba(168, 216, 168, 0.3);
        }
        .metric-card {
            background: linear-gradient(135deg, #e1f5fe 0%, #f3e5f5 100%);
            color: #2c3e50;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.8rem 0;
            box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.6);
        }
        .credit-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 18px;
            margin: 1rem;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .credit-card::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.1);
            transform: rotate(45deg);
        }
        
        /* Enhanced Button Styling - Fixed visibility */
        .stButton > button {
            background: linear-gradient(135deg, #a8d8a8 0%, #b8e6b8 50%, #c8f4c8 100%) !important;
            color: #2c5f2d !important;
            border: 2px solid #90c690 !important;
            border-radius: 25px !important;
            padding: 0.75rem 2rem !important;
            font-weight: bold !important;
            font-size: 1.1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(168, 216, 168, 0.3) !important;
            text-shadow: none !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #90c690 0%, #a8d8a8 50%, #b8e6b8 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(168, 216, 168, 0.4) !important;
            border-color: #7fb67f !important;
        }
        .stButton > button:active {
            transform: translateY(0px) !important;
        }
        .stButton > button:focus {
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(168, 216, 168, 0.3) !important;
        }
        
        /* Message Styling */
        .success-message {
            background: linear-gradient(135deg, #d4edda 0%, #e8f5e9 100%);
            color: #155724;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 4px solid #28a745;
            box-shadow: 0 3px 10px rgba(40, 167, 69, 0.1);
        }
        .warning-message {
            background: linear-gradient(135deg, #f8d7da 0%, #fdeaea 100%);
            color: #721c24;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 4px solid #dc3545;
            box-shadow: 0 3px 10px rgba(220, 53, 69, 0.1);
        }
        
        /* Enhanced Input Styling */
        .stSelectbox > div > div {
            border-radius: 8px;
            border: 1px solid #d1ecf1;
        }
        .stNumberInput > div > div > input {
            border-radius: 8px;
            border: 1px solid #d1ecf1;
        }
        .stSlider > div > div > div {
            color: #6c757d;
        }
        
        /* Sidebar Styling */
        .css-1d391kg {
            background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
        }
        
        /* Form Styling */
        .stForm {
            border: 1px solid rgba(168, 216, 168, 0.2);
            border-radius: 15px;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.5);
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="main-header">
        <h1>💳 Credit Card Eligibility Checker</h1>
        <p style="font-size: 1.2rem; margin: 0.5rem 0;">Professional financial assessment tool to determine your credit card eligibility</p>
        <p style="font-size: 1rem; opacity: 0.8;">✨ Get instant results with personalized recommendations ✨</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.header("📊 Quick Facts")
    st.info("""
    **Minimum Requirements:**
    - Age: 18+ years
    - Income: ₹20,000/month
    - Employment: Salaried/Self-employed
    """)
    
    st.header("💡 Tips")
    st.success("""
    **Improve Your Eligibility:**
    - Maintain stable income
    - Build credit history
    - Keep debt-to-income ratio low
    """)
    
    st.header("🏦 Card Types")
    st.write("""
    - **Basic**: ₹20K+ income
    - **Silver**: ₹50K+ income  
    - **Gold**: ₹100K+ income
    - **Platinum**: ₹200K+ income
    """)

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Personal Information")
    
    # Input Form
    with st.form("eligibility_form"):
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            age = st.number_input(
                "🎂 Age",
                min_value=18,
                max_value=80,
                value=30,
                step=1,
                help="Enter your current age (18-80 years)"
            )
            
            employment_status = st.selectbox(
                "💼 Employment Status",
                ["Salaried", "Self-Employed", "Student", "Unemployed", "Retired"],
                help="Select your current employment type"
            )
            
            experience = st.slider(
                "📈 Work Experience (years)",
                min_value=0,
                max_value=40,
                value=5,
                help="Total years of work experience"
            )
        
        with input_col2:
            income = st.number_input(
                "💰 Monthly Income (₹)",
                min_value=0,
                max_value=1000000,
                value=50000,
                step=5000,
                help="Enter your monthly gross income"
            )
            
            existing_cards = st.number_input(
                "💳 Existing Credit Cards",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Number of credit cards you currently have"
            )
            
            city_tier = st.selectbox(
                "🏙️ City Tier",
                ["Tier 1 (Metro)", "Tier 2", "Tier 3", "Rural"],
                help="Select your city classification"
            )
        
        # Enhanced submit button
        submitted = st.form_submit_button(
            "🔍 Check Eligibility & Get Recommendations",
            use_container_width=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Quick Calculator")
    
    if age and income:
        # Calculate credit score
        credit_score = calculate_credit_score(age, income, employment_status, experience, existing_cards, city_tier)
        
        # Credit Score Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=credit_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Credit Score"},
            delta={'reference': 700},
            gauge={
                'axis': {'range': [None, 850]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 550], 'color': "lightgray"},
                    {'range': [550, 650], 'color': "yellow"},
                    {'range': [650, 750], 'color': "lightgreen"},
                    {'range': [750, 850], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 700
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Score interpretation
        if credit_score >= 750:
            st.success("🌟 Excellent Credit Score!")
        elif credit_score >= 650:
            st.warning("✅ Good Credit Score")
        elif credit_score >= 550:
            st.warning("⚠️ Fair Credit Score")
        else:
            st.error("❌ Poor Credit Score")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Results Section
if submitted:
    # Get eligibility results
    eligible, message, tier = check_eligibility(age, income, employment_status, experience, existing_cards, city_tier)
    
    st.markdown("---")
    
    if eligible:
        st.markdown(f"""
            <div class="success-message">
                <h3>🎉 Congratulations! You're Eligible!</h3>
                <p>{message}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Card Recommendations
        recommendations = get_card_recommendations(income, credit_score, tier)
        
        st.subheader("💳 Recommended Credit Cards")
        
        card_cols = st.columns(len(recommendations))
        for i, card in enumerate(recommendations):
            with card_cols[i]:
                st.markdown(f"""
                    <div class="credit-card">
                        <h4>{card['name']}</h4>
                        <p><strong>Annual Fee:</strong> {card['fee']}</p>
                        <p><strong>Credit Limit:</strong> {card['limit']}</p>
                        <p><strong>Rewards:</strong> {card['rewards']}</p>
                        <p><strong>Approval Chance:</strong> {card['approval_chance']}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Financial Analysis
        st.subheader("📊 Financial Analysis")
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50;">Credit Score</h4>
                    <h2 style="margin: 0 0 0.2rem 0; color: #2c3e50; font-size: 2.5rem;">{credit_score}</h2>
                    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">Out of 850</p>
                </div>
            """, unsafe_allow_html=True)
        
        with analysis_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50;">Debt-to-Income</h4>
                    <h2 style="margin: 0 0 0.2rem 0; color: #2c3e50; font-size: 2.5rem;">{min(existing_cards * 5000 / income * 100, 100):.1f}%</h2>
                    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">Recommended: &lt;30%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with analysis_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50;">Card Tier</h4>
                    <h2 style="margin: 0 0 0.2rem 0; color: #2c3e50; font-size: 2.5rem;">{tier}</h2>
                    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">Eligibility Level</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Financial Tips
        tips = get_financial_tips(credit_score, income, existing_cards)
        
        st.subheader("💡 Personalized Financial Tips")
        for tip in tips:
            st.info(f"✅ {tip}")
            
    else:
        st.markdown(f"""
            <div class="warning-message">
                <h3>❌ Not Eligible</h3>
                <p>{message}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Improvement suggestions
        st.subheader("🎯 How to Improve Your Eligibility")
        
        improvement_tips = []
        if age < 18:
            improvement_tips.append("Wait until you turn 18 years old")
        if income < 20000:
            improvement_tips.append(f"Increase your monthly income to at least ₹20,000 (currently ₹{income:,})")
        if employment_status not in ['Salaried', 'Self-Employed']:
            improvement_tips.append("Obtain stable employment (Salaried or Self-Employed)")
        
        for tip in improvement_tips:
            st.warning(f"📈 {tip}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>💡 This tool provides estimated eligibility based on basic criteria. 
        Actual approval depends on additional factors like credit history, existing debts, and bank policies.</p>
        <p>🔒 Your data is processed locally and not stored anywhere.</p>
    </div>
""", unsafe_allow_html=True)
