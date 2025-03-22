from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(texts):
    """Convert text into TF-IDF feature vectors"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer
