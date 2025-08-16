
import re
from typing import List
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

_stop = set(stopwords.words('english'))
_sent = SentimentIntensityAnalyzer()

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return [t for t in word_tokenize(s) if t.isalnum() and t not in _stop]

def sentiment(s: str) -> dict:
    return _sent.polarity_scores(s)
