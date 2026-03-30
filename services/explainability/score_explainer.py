import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from services.matching.semantic_scorer import get_model

def get_semantic_matched_missing(candidate_skills: set, req_skills: set):
    if not req_skills:
        return [], []
    if not candidate_skills:
        return [], list(req_skills)
        
    sentence_model = get_model()
    
    req_list = list(req_skills)
    cand_list = list(candidate_skills)
    
    req_embeddings = sentence_model.encode(req_list)
    cand_embeddings = sentence_model.encode(cand_list)
    
    sim_matrix = cosine_similarity(req_embeddings, cand_embeddings)
    
    matched = []
    missing = []
    
    for i, req in enumerate(req_list):
        max_sim = float(np.max(sim_matrix[i]))
        # We lower the threshold slightly to catch specific subsets like "pandas" vs "python libraries"
        if max_sim >= 0.50:
            matched.append(req)
        else:
            missing.append(req)
            
    return matched, missing

def generate_explanation(parsed_resume: dict, parsed_jd: dict, scoring_result: dict) -> dict:
    """
    Generates explainable output dynamically computing strengths, weak areas,
    and missing skill gaps for a specific candidate regarding a specific role.
    """
    candidate_skills = set([s.lower() for s in parsed_resume.get("skills", [])])
    req_skills = set([s.lower() for s in parsed_jd.get("required_skills", [])])
    
    # 🌟 UPGRADED: Semantic Skill Mapping (Layer 3 Expansion)
    # This prevents the system from failing to recognize that "python libraries" covers "pandas".
    matched_skills, missing_skills = get_semantic_matched_missing(candidate_skills, req_skills)
    
    strengths = []
    weak_areas = []
    warnings = []
    rejection_reasons = []
    
    # Check for JD Extraction failures
    jd_extraction_failed = False
    if not req_skills:
        warnings.append("Requirements did not contain any recognized skills from our taxonomy. Skill matching score defaulted to 0.")
        jd_extraction_failed = True

    # Analyze Skills Match (Only if JD skills exist)
    if not jd_extraction_failed:
        if len(matched_skills) == len(req_skills) and len(req_skills) > 0:
            strengths.append(f"Candidate possesses all {len(req_skills)} required skills.")
        elif len(matched_skills) >= (len(req_skills) * 0.7):
            strengths.append(f"Strong technical match: {len(matched_skills)}/{len(req_skills)} required skills met.")
            weak_areas.append(f"Missing {len(missing_skills)} required skills")
        elif len(matched_skills) > 0:
            weak_areas.append(f"Lacks {len(missing_skills)} critical skills. Only {len(matched_skills)}/{len(req_skills)} matched.")
        else:
            weak_areas.append(f"Candidate possesses 0 of the {len(req_skills)} required skills.")
            rejection_reasons.append("Does not meet any of the required skills.")
        
    # Context & Experience Analysis (Using Raw Scores)
    exp_raw = scoring_result["raw_scores"].get("experience", 0.0)
    sem_raw = scoring_result["raw_scores"].get("semantic", 0.0)
    role_raw = scoring_result["raw_scores"].get("role_alignment", 0.0)
    
    if exp_raw > 0.65:
        strengths.append(f"High experience overlap ({exp_raw*100:.1f}% context match) with job duties.")
    elif exp_raw < 0.35:
        weak_areas.append(f"Experience text highly disconnected from the job role ({exp_raw*100:.1f}% context match). Check for role mismatch (e.g., non-technical vs technical).")

    if sem_raw > 0.65:
        strengths.append(f"Overall resume language naturally matches the job context ({sem_raw*100:.1f}% overall similarity).")
    elif sem_raw < 0.35:
        weak_areas.append(f"Overall resume tone and domain vocabulary diverges significantly from the job description ({sem_raw*100:.1f}% overall similarity).")

    if role_raw > 0.7:
        strengths.append("High alignment with the specific job role.")
    elif role_raw < 0.3:
        rejection_reasons.append("Resume shows very little alignment with the stated job role.")
        
    return {
        "job_description_skill_extraction_failed": jd_extraction_failed,
        "missing_skills": missing_skills,
        "matched_skills": matched_skills,
        "strengths": strengths,
        "weak_areas": weak_areas,
        "rejection_reasons": rejection_reasons,
        "warnings": warnings
    }
