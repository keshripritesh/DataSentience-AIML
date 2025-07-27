import re

def clean_hashtag(tag):
    tag = tag.lower()
    tag = re.sub(r'[^a-z0-9#]', '', tag)
    return tag
