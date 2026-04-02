import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)


def preprocess_sentences(sentences):
    return [preprocess_text(s) for s in sentences]


def tfidf_summarize(article, n=3):
    sentences = sent_tokenize(article)

    # Remove short sentences under 8 words
    sentences = [s for s in sentences if len(s.split()) > 8]

    if len(sentences) <= n:
        return ' '.join(sentences)

    cleaned = preprocess_sentences(sentences)
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        return ' '.join(sentences[:n])

    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # Boost first and last sentence (inverted pyramid structure)
    scores[0]  *= 1.3
    scores[-1] *= 1.2

    top_indices = sorted(np.argsort(scores)[-n:])
    return ' '.join([sentences[i] for i in top_indices])


def textrank_summarize(article, n=3):
    sentences = sent_tokenize(article)

    # Remove short sentences under 8 words
    sentences = [s for s in sentences if len(s.split()) > 8]

    if len(sentences) <= n:
        return ' '.join(sentences)

    cleaned = preprocess_sentences(sentences)
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned).toarray()
    except ValueError:
        return ' '.join(sentences[:n])

    sim_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph, max_iter=200)

    # Boost first and last sentence — scores is a dict in TextRank
    first_key = list(scores.keys())[0]
    last_key  = list(scores.keys())[-1]
    scores[first_key] *= 1.3
    scores[last_key]  *= 1.2

    ranked_indices = sorted(scores, key=scores.get, reverse=True)[:n]
    top_indices = sorted(ranked_indices)
    return ' '.join([sentences[i] for i in top_indices])


def frequency_summarize(article, n=3):
    sentences = sent_tokenize(article)

    # Remove short sentences under 8 words
    sentences = [s for s in sentences if len(s.split()) > 8]

    if len(sentences) <= n:
        return ' '.join(sentences)

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(article.lower())
    freq_table = {}
    for word in words:
        word = re.sub(r'[^a-zA-Z]', '', word)
        if word and word not in stop_words:
            freq_table[word] = freq_table.get(word, 0) + 1

    max_freq = max(freq_table.values(), default=1)
    freq_table = {w: f / max_freq for w, f in freq_table.items()}

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq_table:
                sentence_scores[i] = sentence_scores.get(i, 0) + freq_table[word]

    # Boost first and last sentence
    if 0 in sentence_scores:
        sentence_scores[0] *= 1.3
    if len(sentences) - 1 in sentence_scores:
        sentence_scores[len(sentences) - 1] *= 1.2

    top_indices = sorted(
        sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:n]
    )
    return ' '.join([sentences[i] for i in top_indices])


def summarize(article, method='tfidf', n=3):
    article = article.strip()
    if not article:
        return "No article provided."
    method = method.lower()
    if method == 'tfidf':
        return tfidf_summarize(article, n)
    elif method == 'textrank':
        return textrank_summarize(article, n)
    elif method == 'frequency':
        return frequency_summarize(article, n)
    else:
        raise ValueError(f"Unknown method: '{method}'")