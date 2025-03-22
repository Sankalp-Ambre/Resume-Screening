import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import clean_text  # Now it should work!

from src.preprocess import clean_text
from src.feature_extraction import extract_features
from src.similarity import compute_similarity
from src.ranking import rank_resumes

# Load resumes from data/resumes/
resume_folder = "data/resumes"
resume_files = os.listdir(resume_folder)

# Read resumes
resumes = {}
for file in resume_files:
    with open(os.path.join(resume_folder, file), "r", encoding="utf-8") as f:
        resumes[file] = f.read()

# Read job description
with open("data/job_description.txt", "r", encoding="utf-8") as f:
    job_description = f.read()

# Preprocess text
processed_resumes = {name: clean_text(text) for name, text in resumes.items()}
processed_job_desc = clean_text(job_description)

# Convert to TF-IDF vectors
texts = list(processed_resumes.values()) + [processed_job_desc]
tfidf_matrix, vectorizer = extract_features(texts)

# Compute similarity
similarity_scores = compute_similarity(tfidf_matrix)

# Rank resumes
ranked_resumes = rank_resumes(list(processed_resumes.keys()), similarity_scores)

# Print results
print("\nTop Matching Resumes:")
for rank, (name, score) in enumerate(ranked_resumes, start=1):
    print(f"{rank}. {name} - Similarity Score: {score:.2f}")
