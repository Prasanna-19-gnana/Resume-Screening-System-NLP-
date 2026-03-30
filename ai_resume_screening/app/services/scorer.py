import logging
from typing import Dict, Any, List, Tuple
from .semantic_matcher import get_sectioned_semantic_similarity
from .skill_extractor import SkillExtractor
from .role_classifier import role_classifier_svc

logger = logging.getLogger(__name__)

class AIScorer:
    def __init__(self):
        self.skill_ext = SkillExtractor()
        
    def _get_recommendation_tier(self, score: float) -> str:
        s = score * 100
        if s >= 80: return "Strong Match"
        elif s >= 65: return "Moderate Match"
        elif s >= 45: return "Weak Match"
        else: return "Poor Match"
        
    def score_candidate(self, parsed_resume_dict: dict, job_role: str, requirements: str, job_description: str, resume_semantic_text: str) -> Dict[str, Any]:
        """
        Executes the intelligent dynamic 3-layer hybrid scoring matrix using Section-Aware Embeddings!
        """
        logger.info(f"Scoring candidate against role: {job_role}")
        
        # 1. Multi-Label Role Classifier Component
        role_probs = role_classifier_svc.predict_resume_role_probabilities(resume_semantic_text)
        align_score = role_classifier_svc.calculate_role_alignment(role_probs, job_role)
        pred_role_name = list(role_probs.keys())[0] if role_probs else "Unknown"
        
        # 2. Skill Extraction and Expansion Ontology Component
        c_skills = self.skill_ext.extract_skills(resume_semantic_text)
        req_explicit = self.skill_ext.extract_from_requirements(requirements)
        jd_implicit = self.skill_ext.extract_skills(job_description)
        req_comp = list(set(req_explicit + jd_implicit))
        
        matched, missing, extra, skill_score = self.skill_ext.evaluate_skills(c_skills, req_comp)
        
        # 3. Section-Aware Semantic Similarities
        # Instead of embedding the full text against the full JD trivially, we pass segmented dicts
        sem_metrics = get_sectioned_semantic_similarity(
            resume_sections=parsed_resume_dict,
            job_role=job_role,
            requirements=requirements,
            job_description=job_description,
            extracted_jd_skills=req_comp
        )
        semantic_score = sem_metrics["semantic_similarity_score"]
        
        # 4. Suggested Final Weighting: Sem 40%, Skill 35%, Role 25%
        sem_wt = 0.40
        sk_wt = 0.35
        rl_wt = 0.25
        
        final_score_raw = (sem_wt * semantic_score) + (sk_wt * skill_score) + (rl_wt * align_score)
        final_score = round(max(0.0, min(1.0, final_score_raw)), 4)
        
        return {
            "predicted_resume_role": pred_role_name,
            "role_probabilities": role_probs,
            "role_alignment_score": round(align_score, 4),
            
            # Map detailed semantic dict seamlessly
            "skills_similarity": sem_metrics["skills_similarity"],
            "projects_similarity": sem_metrics["projects_similarity"],
            "experience_similarity": sem_metrics["experience_similarity"],
            "semantic_similarity_score": round(semantic_score, 4),
            
            "skill_match_score": round(skill_score, 4),
            "matched_skills": matched,
            "missing_skills": missing,
            "extra_skills": extra,
            
            "final_score": final_score,
            "recommendation": self._get_recommendation_tier(final_score)
        }

# Expose singleton
scorer_service = AIScorer()
