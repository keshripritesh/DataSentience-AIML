import snscrape.modules.twitter as sntwitter

def get_tweets(query, limit=50):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit: break
        tweets.append(tweet.content)
    return tweets
