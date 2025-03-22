from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(tfidf_matrix):
    """Compute cosine similarity between job description and resumes"""
    job_desc_vector = tfidf_matrix[-1]  # Last row is the job description
    resume_vectors = tfidf_matrix[:-1]  # All other rows are resumes
    similarity_scores = cosine_similarity(job_desc_vector, resume_vectors).flatten()
    return similarity_scores
