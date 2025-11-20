import pdfplumber
import re
from collections import Counter

# -----------------------------
# PDF TEXT EXTRACTOR
# -----------------------------
def extract_text_from_pdf(file_obj):
    """
    Reads PDF file using pdfplumber and returns extracted text.
    file_obj can be request.FILES['file']
    """
    text = ""
    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print("PDF extraction error:", e)
        return ""

    return text


# -----------------------------
# TOPIC EXTRACTOR
# -----------------------------
STOPWORDS = set([
    "the","is","and","of","to","a","in","for","on","with","that","this","are",
    "as","by","be","it","an","or","from","which","at","these","will","have","has"
])

def extract_topics(text, top_n=8):
    """
    Extract simple topic keywords from text.
    """
    if not text:
        return []

    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    words = [w for w in cleaned.split() if w and w not in STOPWORDS and len(w) > 3]

    if not words:
        return []

    freq = Counter(words)
    topics = [t for t,_ in freq.most_common(top_n)]
    return topics
