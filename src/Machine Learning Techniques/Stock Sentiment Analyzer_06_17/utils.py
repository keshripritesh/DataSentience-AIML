def predict_market_impact(sentiment_result):
    pos = sentiment_result['positive']
    neg = sentiment_result['negative']

    if pos > 70:
        return "Likely Bullish 📈"
    elif neg > 70:
        return "Likely Bearish 📉"
    else:
        return "Neutral/Uncertain 🤔"
