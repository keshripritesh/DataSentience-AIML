def check_eligibility(age, income, employment_status, experience=0, existing_cards=0, city_tier="Tier 1 (Metro)"):
    """
    Enhanced eligibility checker with multiple factors
    
    Args:
        age: Applicant's age
        income: Monthly income in INR
        employment_status: Employment type
        experience: Work experience in years
        existing_cards: Number of existing credit cards
        city_tier: City classification
    
    Returns:
        tuple: (eligible, message, tier)
    """
    
    # Basic eligibility checks
    if age < 18:
        return False, "❌ Age must be 18 or above to apply for a credit card.", "None"
    
    if age > 65:
        return False, "❌ Age limit exceeded. Most banks have an upper age limit of 65 years.", "None"
    
    if income < 15000:
        return False, f"❌ Monthly income of ₹{income:,} is below minimum requirement of ₹15,000.", "None"
    
    if employment_status.lower() not in ['salaried', 'self-employed']:
        return False, f"❌ Employment status '{employment_status}' is not eligible. Only Salaried or Self-employed individuals qualify.", "None"
    
    # Determine card tier based on income and other factors
    if income >= 200000:
        tier = "Platinum"
        message = f"🌟 Excellent! You qualify for Platinum tier cards with premium benefits."
    elif income >= 100000:
        tier = "Gold"
        message = f"✨ Great! You qualify for Gold tier cards with enhanced rewards."
    elif income >= 50000:
        tier = "Silver"
        message = f"👍 Good! You qualify for Silver tier cards with good benefits."
    elif income >= 20000:
        tier = "Basic"
        message = f"✅ You qualify for Basic tier credit cards."
    else:
        tier = "Entry"
        message = f"📝 You qualify for entry-level cards with basic features."
    
    # Add experience bonus
    if experience >= 5:
        message += f" Your {experience} years of experience strengthens your application."
    elif experience >= 2:
        message += f" Your {experience} years of experience is favorable for approval."
    
    # Existing cards impact
    if existing_cards > 3:
        message += f" ⚠️ Having {existing_cards} existing cards might require additional verification."
    elif existing_cards > 0:
        message += f" Your existing {existing_cards} card(s) show good credit management."
    
    return True, message, tier


def calculate_credit_score(age, income, employment_status, experience=0, existing_cards=0, city_tier="Tier 1 (Metro)"):
    """
    Calculate estimated credit score based on provided information
    
    Returns:
        int: Estimated credit score (300-850)
    """
    
    # Base score
    score = 300
    
    # Age factor (25-45 is optimal)
    if 25 <= age <= 45:
        score += 100
    elif 18 <= age <= 24 or 46 <= age <= 60:
        score += 80
    else:
        score += 60
    
    # Income factor
    if income >= 200000:
        score += 150
    elif income >= 100000:
        score += 120
    elif income >= 50000:
        score += 100
    elif income >= 30000:
        score += 80
    elif income >= 20000:
        score += 60
    else:
        score += 40
    
    # Employment factor
    if employment_status.lower() == 'salaried':
        score += 80
    elif employment_status.lower() == 'self-employed':
        score += 70
    else:
        score += 30
    
    # Experience factor
    if experience >= 10:
        score += 80
    elif experience >= 5:
        score += 60
    elif experience >= 2:
        score += 40
    else:
        score += 20
    
    # Existing cards factor (0-2 cards is optimal)
    if existing_cards == 0:
        score += 40  # No credit history penalty
    elif 1 <= existing_cards <= 2:
        score += 70  # Optimal number
    elif 3 <= existing_cards <= 4:
        score += 50  # Manageable
    else:
        score += 20  # Too many cards
    
    # City tier factor
    city_scores = {
        "Tier 1 (Metro)": 60,
        "Tier 2": 50,
        "Tier 3": 40,
        "Rural": 30
    }
    score += city_scores.get(city_tier, 40)
    
    # Cap the score at 850
    return min(score, 850)


def get_card_recommendations(income, credit_score, tier):
    """
    Get personalized credit card recommendations
    
    Returns:
        list: List of recommended cards with details
    """
    
    recommendations = []
    
    if tier == "Platinum":
        recommendations.extend([
            {
                "name": "Platinum Rewards Card",
                "fee": "₹5,000/year",
                "limit": "₹8-15 lakhs",
                "rewards": "5X points on dining, 3X on fuel",
                "approval_chance": "85%"
            },
            {
                "name": "Premium Travel Card",
                "fee": "₹10,000/year",
                "limit": "₹10-20 lakhs",
                "rewards": "Airport lounge access, travel insurance",
                "approval_chance": "80%"
            }
        ])
    
    elif tier == "Gold":
        recommendations.extend([
            {
                "name": "Gold Cashback Card",
                "fee": "₹2,500/year",
                "limit": "₹3-8 lakhs",
                "rewards": "5% cashback on groceries, 2% on fuel",
                "approval_chance": "90%"
            },
            {
                "name": "Gold Rewards Card",
                "fee": "₹1,999/year",
                "limit": "₹2-6 lakhs",
                "rewards": "2X points on all purchases",
                "approval_chance": "88%"
            }
        ])
    
    elif tier == "Silver":
        recommendations.extend([
            {
                "name": "Silver Plus Card",
                "fee": "₹999/year",
                "limit": "₹1-3 lakhs",
                "rewards": "1% cashback on all spends",
                "approval_chance": "92%"
            },
            {
                "name": "Lifestyle Card",
                "fee": "₹750/year",
                "limit": "₹75K-2 lakhs",
                "rewards": "2X points on shopping and dining",
                "approval_chance": "90%"
            }
        ])
    
    else:  # Basic or Entry
        recommendations.extend([
            {
                "name": "Basic Starter Card",
                "fee": "Free for first year",
                "limit": "₹25K-1 lakh",
                "rewards": "Welcome bonus ₹500",
                "approval_chance": "95%"
            },
            {
                "name": "Student Card",
                "fee": "₹199/year",
                "limit": "₹15K-50K",
                "rewards": "1X points on all spends",
                "approval_chance": "93%"
            }
        ])
    
    return recommendations


def get_financial_tips(credit_score, income, existing_cards):
    """
    Generate personalized financial tips
    
    Returns:
        list: List of financial improvement tips
    """
    
    tips = []
    
    # Credit score based tips
    if credit_score < 600:
        tips.extend([
            "Build credit history by using a secured credit card",
            "Pay all bills on time to improve payment history",
            "Keep credit utilization below 30% of limit"
        ])
    elif credit_score < 700:
        tips.extend([
            "Maintain current payment discipline",
            "Consider increasing credit limit to lower utilization ratio",
            "Avoid applying for multiple cards in short period"
        ])
    else:
        tips.extend([
            "Excellent credit score! You qualify for premium cards",
            "Consider cards with higher rewards for your spending pattern",
            "You may negotiate for better terms with banks"
        ])
    
    # Income based tips
    if income < 50000:
        tips.append("Focus on increasing income through skill development or side income")
    elif income < 100000:
        tips.append("Consider investment options to grow your wealth")
    else:
        tips.append("Explore premium banking services and wealth management")
    
    # Existing cards tips
    if existing_cards == 0:
        tips.append("Start with one card to build credit history")
    elif existing_cards > 3:
        tips.append("Consider consolidating cards to manage them better")
    
    # General tips
    tips.extend([
        "Set up automatic payments to never miss due dates",
        "Review your credit report annually for accuracy",
        "Use credit cards for planned purchases, not impulse buying"
    ])
    
    return tips[:6]  # Return top 6 tips
