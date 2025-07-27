import re

def clean_text(text):
    # Keep links but lowercase and remove symbols
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()
