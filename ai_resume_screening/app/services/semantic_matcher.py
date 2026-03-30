import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SemanticMatcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        
    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading SentenceTransformer: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
    def _compute_cosine(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            embeddings = self.model.encode([text1, text2])
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            return 0.0
    
    def get_sectioned_semantic_similarity(
        self, 
        resume_sections: dict, 
        job_role: str,
        requirements: str,
        job_description: str,
        extracted_jd_skills: list
    ) -> dict:
        """
        Computes deeply segmented semantic similarities per block and redistributes
        weights actively if sections are entirely blank (e.g. no projects listed).
        """
        self._load_model()
        
        skills_text = resume_sections.get("skills", "").strip()
        projects_text = resume_sections.get("projects", "").strip()
        experience_text = resume_sections.get("experience", "").strip()
        
        # Determine lengths
        len_s = len(skills_text.split())
        len_p = len(projects_text.split())
        len_e = len(experience_text.split())
        
        logger.info(f"Section Lengths - Skills: {len_s}, Projects: {len_p}, Exp: {len_e}")
        
        # Compute A) Skills vs (Requirements + Extracted JD Skills)
        target_skills_text = f"{requirements} {' '.join(extracted_jd_skills)}".strip()
        if not target_skills_text: target_skills_text = job_description
        
        sim_skills = self._compute_cosine(skills_text, target_skills_text)
        
        # Compute B) Projects vs (Role + JD)
        target_projects_text = f"Role: {job_role}. {job_description}"
        sim_projects = self._compute_cosine(projects_text, target_projects_text)
        
        # Compute C) Experience vs (Role + JD)
        target_experience_text = f"Role: {job_role}. {job_description}"
        sim_experience = self._compute_cosine(experience_text, target_experience_text)
        
        logger.info(f"Raw Similarities - Skills: {sim_skills:.3f}, Projects: {sim_projects:.3f}, Experience: {sim_experience:.3f}")
        
        # Default Weights
        w_skills = 0.45
        w_projects = 0.35
        w_experience = 0.20
        
        # Dynamic Weight Redistribution for Missing Sections
        # We assume a section is truly "missing" if length < 5 tokens
        available_w = 0.0
        
        if len_s >= 5: available_w += w_skills
        if len_p >= 5: available_w += w_projects
        if len_e >= 5: available_w += w_experience
        
        if available_w == 0.0:
            logger.warning("No valid technical sections found in resume to embed.")
            semantic_score = 0.0
        else:
            # Rebalance existing valid weights to sum to 1.0!
            act_w_skills = w_skills / available_w if len_s >= 5 else 0.0
            act_w_projects = w_projects / available_w if len_p >= 5 else 0.0
            act_w_experience = w_experience / available_w if len_e >= 5 else 0.0
            
            semantic_score = (act_w_skills * sim_skills) + (act_w_projects * sim_projects) + (act_w_experience * sim_experience)
            logger.info(f"Redistributed Weights - Skills: {act_w_skills:.2f}, Proj: {act_w_projects:.2f}, Exp: {act_w_experience:.2f}")

        logger.info(f"Final Weighted Semantic Score: {semantic_score:.4f}")
        
        return {
            "skills_similarity": round(sim_skills, 4),
            "projects_similarity": round(sim_projects, 4),
            "experience_similarity": round(sim_experience, 4),
            "semantic_similarity_score": round(semantic_score, 4)
        }

# Export a singleton instance
semantic_matcher = SemanticMatcher()

def get_sectioned_semantic_similarity(resume_sections: dict, job_role: str, requirements: str, job_description: str, extracted_jd_skills: list) -> dict:
    return semantic_matcher.get_sectioned_semantic_similarity(resume_sections, job_role, requirements, job_description, extracted_jd_skills)
