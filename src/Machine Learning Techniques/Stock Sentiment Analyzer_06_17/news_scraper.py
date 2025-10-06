import requests

def get_news_headlines(stock_name, api_key):
    url =  f"https://newsapi.org/v2/everything?q={stock_name}&apiKey={api_key}"
    print(url)
    res = requests.get(url).json()
    return [article['title'] for article in res['articles'][:10]]
