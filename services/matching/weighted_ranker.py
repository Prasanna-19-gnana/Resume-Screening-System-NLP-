import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from services.matching.semantic_scorer import get_model

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CLF_PATH = os.path.join(BASE_DIR, "models/rf_classifier.pkl")
REG_PATH = os.path.join(BASE_DIR, "models/rf_regressor.pkl")

# Safely load models at module level. If missing, we fallback automatically.
rf_classifier = None
rf_regressor = None

try:
    if os.path.exists(CLF_PATH): rf_classifier = joblib.load(CLF_PATH)
    if os.path.exists(REG_PATH): rf_regressor = joblib.load(REG_PATH)
except Exception as e:
    print(f"Warning: Failed to load ML models: {e}")

def role_classifier(job_role: str) -> str:
    """ Layer 1: Resume Understanding (Job Role classification mapping) """
    role = job_role.lower()
    if any(k in role for k in ["marketing", "sales", "pr", "communications"]):
        return "marketing"
    elif any(k in role for k in ["data", "analyst", "scientist", "ai", "machine learning"]):
        return "data"
    elif any(k in role for k in ["developer", "engineer", "programmer", "backend", "frontend", "fullstack", "software"]):
        return "developer"
    return "general"

def skill_match_with_expansion(candidate_skills: list, req_skills: list) -> float:
    """ Layer 3: Skill Matching (WITH EXPANSION)
    Computes semantic similarity matrix between candidate skills and required skills
    to reward partial matches (e.g., matching 'machine learning' dynamically rewards 'tensorflow')
    """
    if not req_skills:
        return 0.0
    if not candidate_skills:
        return 0.0
        
    sentence_model = get_model()
    
    # Semantic Expansion Embeddings Layer
    req_embeddings = sentence_model.encode(req_skills)
    cand_embeddings = sentence_model.encode(candidate_skills)
    
    sim_matrix = cosine_similarity(req_embeddings, cand_embeddings)
    
    total_score = 0.0
    
    for i in range(len(req_skills)):
        max_sim = float(np.max(sim_matrix[i]))
        if max_sim >= 0.85:
            total_score += 1.0  # Exact or highly synonymous match
        elif max_sim >= 0.50:
            total_score += max_sim  # Partial taxonomic match (e.g., 'machine learning' -> 'tensorflow')
        else:
            total_score += 0.0  # No structural relationship detected
            
    return total_score / len(req_skills)

def rank_candidate(parsed_resume: dict, parsed_jd: dict, scorer_mode: str = "compare") -> dict:
    """
    Ranks candidate dynamically using structured metrics according to EXACT formula:
    Final Score = 0.5 * semantic_similarity + 0.3 * skill_match_score + 0.2 * role_alignment
    """
    candidate_skills = parsed_resume.get("skills", [])
    req_skills = parsed_jd.get("required_skills", [])
    
    # Layer 1 & 2 Metrics
    semantic_raw_score = parsed_resume.get("overall_semantic_similarity", 0.0)
    role_raw_score = parsed_resume.get("role_alignment", 0.0)
    exp_raw_score = parsed_resume.get("experience_relevance", 0.0)
    
    # Layer 3 Metric
    skills_raw_score = skill_match_with_expansion(candidate_skills, req_skills)
    
    # FINAL SCORE (DO THIS EXACTLY)
    skills_weighted = skills_raw_score * 0.30
    semantic_weighted = semantic_raw_score * 0.50
    role_weighted = role_raw_score * 0.20
    
    baseline_score = skills_weighted + semantic_weighted + role_weighted
    
    result = {
        "scorer_mode": scorer_mode,
        "baseline_score": round(baseline_score, 4),
        "raw_scores": {
            "skills": round(skills_raw_score, 4), 
            "experience": round(exp_raw_score, 4),  # Preserved for API mapping compat
            "role_alignment": round(role_raw_score, 4), 
            "semantic": round(semantic_raw_score, 4)
        },
        "weighted_scores": {
            "skills": round(skills_weighted, 4), 
            "role_alignment": round(role_weighted, 4), 
            "semantic": round(semantic_weighted, 4)
        }
    }
    
    # ML compatibility payload
    if scorer_mode in ["ml", "compare"]:
        req_skills_set = set(s.lower() for s in req_skills)
        matched = set(s.lower() for s in candidate_skills).intersection(req_skills_set)
        
        req_coverage = len(matched) / len(req_skills_set) if len(req_skills_set) > 0 else 0.0
        missing_req = len(req_skills_set) - len(matched)
        resume_words = len(parsed_resume.get("resume_text", "").split())
        
        feature_dict = {
            "req_skill_coverage": [float(req_coverage)],
            "pref_skill_coverage": [0.0],
            "total_matched_skills": [float(len(matched))],
            "missing_req_skills": [float(missing_req)],
            "semantic_similarity": [float(semantic_raw_score)],
            "resume_word_count": [float(resume_words)]
        }
        df_features = pd.DataFrame(feature_dict)
        
        ml_score = 0.0
        fit_label = "unknown"
        
        if rf_regressor and rf_classifier:
            try:
                predicted_val = rf_regressor.predict(df_features)[0]
                ml_score = predicted_val / 100.0 if predicted_val > 1.0 else predicted_val
                fit_label = rf_classifier.predict(df_features)[0]
            except Exception as e:
                pass
                
        result["ml_score"] = round(ml_score, 4)
        result["fit_label_prediction"] = fit_label
        
        if scorer_mode == "ml":
            result["final_score"] = result.get("ml_score", baseline_score)
        else:
            result["final_score"] = baseline_score # use new exact baseline score
            
    else:
        result["final_score"] = baseline_score
        
    return result
