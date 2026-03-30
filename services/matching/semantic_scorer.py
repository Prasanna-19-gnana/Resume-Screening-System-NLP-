from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load local pre-trained model on first import. 
# 'all-MiniLM-L6-v2' is small, fast, and highly effective for general semantic similarity.
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Computes cosine similarity between two text blocks using SentenceTransformers.
    Fallback to 0 if text is missing.
    """
    if not text1.strip() or not text2.strip():
        return 0.0
        
    sentence_model = get_model()
    embeddings = sentence_model.encode([text1, text2])
    
    # Cosine similarity yields a 2D matrix
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Ensure range is [0.0, 1.0] to prevent negative scores propagating down
    return max(0.0, min(1.0, float(sim)))

def compute_experience_relevance(exp_text: str, jd_text: str) -> float:
    """
    Calculate semantic distance strictly between the candidate's Experience/Project section
    and the overall Job Description.
    """
    return compute_semantic_similarity(exp_text, jd_text)

def compute_education_relevance(edu_text: str, jd_text: str) -> float:
    """
    Calculate semantic distance between Education and the Job Description.
    While less weight is given to education context, similarity remains a good signal.
    """
    # If the candidate has missing education blocks, fallback to 0.0
    if not edu_text.strip():
        return 0.0
    return compute_semantic_similarity(edu_text, jd_text)
