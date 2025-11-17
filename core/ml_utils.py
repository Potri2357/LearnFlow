import nltk
from nltk.corpus import stopwords
from collections import Counter

def extract_topics(text, top_n=5):
    words = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))

    filtered = [w for w in words if w.isalpha() and w not in stop_words]

    freq = Counter(filtered)
    topics = [word for word, count in freq.most_common(top_n)]
    return topics
