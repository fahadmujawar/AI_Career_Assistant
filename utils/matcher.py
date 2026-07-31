import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "in",
    "on", "at", "to", "for", "with", "by", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "we", "you", "your", "our", "their", "will", "would", "can",
    "could", "should", "may", "might", "must", "have", "has", "had", "do",
    "does", "did", "not", "no", "yes", "such", "than", "into", "about",
    "across", "per", "etc", "including", "e.g", "i.e"
}


def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    return words


def extract_ngrams(words):
    unigrams = [w for w in words if w not in STOPWORDS and len(w) > 2]

    bigrams = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 not in STOPWORDS and w2 not in STOPWORDS:
            bigrams.append(f"{w1} {w2}")

    return unigrams, bigrams


def get_important_keywords(jd_text, top_n=25):
    words = clean_and_tokenize(jd_text)
    unigrams, bigrams = extract_ngrams(words)

    combined = unigrams + bigrams
    counts = Counter(combined)

    most_common = counts.most_common(top_n)
    return [term for term, count in most_common]

# LIMITATION: This matcher does exact string comparison after basic cleaning
# (lowercase + punctuation removal) but does not handle word variants like
# "client" vs "clients", or "manage" vs "managing"/"managed". Properly
# handling this requires stemming (crudely chopping word endings, e.g.
# "clients" -> "client") or lemmatization (using a dictionary/grammar-aware
# approach to find a word's true base/dictionary form, e.g. "managing" ->
# "manage", "better" -> "good"). Lemmatization is more accurate but needs an
# external NLP library (e.g. nltk or spaCy) and a bit more setup.
#
# We're intentionally not solving this here: this keyword matcher is a
# stepping stone toward an LLM-based CV/JD comparison, and an LLM naturally
# understands "client" and "clients" mean the same thing without any special
# handling. Adding stemming/lemmatization now would fix a problem that
# mostly disappears once the LLM-based comparison is built.
def match_keywords(cv_text, jd_text):
    cv_normalized = " ".join(clean_and_tokenize(cv_text))
    keywords = get_important_keywords(jd_text)

    matched = []
    missing = []

    for keyword in keywords:
        if keyword in cv_normalized:
            matched.append(keyword)
        else:
            missing.append(keyword)

    total = len(keywords)
    match_score = round(len(matched) / total * 100, 1) if total > 0 else 0.0

    return {
        "matched": matched,
        "missing": missing,
        "score": match_score
    }
def match_keywords(cv_text, jd_text):
    cv_normalized = " ".join(clean_and_tokenize(cv_text))
    keywords = get_important_keywords(jd_text)

    matched = []
    missing = []

    for keyword in keywords:
        if keyword in cv_normalized:
            matched.append(keyword)
        else:
            missing.append(keyword)

    total = len(keywords)
    match_score = round(len(matched) / total * 100, 1) if total > 0 else 0.0

    return {
        "matched": matched,
        "missing": missing,
        "score": match_score
    }