import re

def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+", '', tweet)
    tweet = re.sub(r"@\w+|#", '', tweet)
    tweet = re.sub(r"[^a-zA-Z\s]", '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet
