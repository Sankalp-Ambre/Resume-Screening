import json

def rank_resumes(resume_names, similarity_scores, output_file="results/ranked_resumes.json"):
    """Rank resumes based on similarity scores and save results"""
    ranked_resumes = sorted(zip(resume_names, similarity_scores), key=lambda x: x[1], reverse=True)
    with open(output_file, "w") as f:
        json.dump(ranked_resumes, f, indent=4)
    return ranked_resumes
