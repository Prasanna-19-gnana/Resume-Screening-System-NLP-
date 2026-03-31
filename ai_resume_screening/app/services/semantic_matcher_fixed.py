"""
FIXED: Semantic Matcher with Proper Text Cleaning
- Use only technical sections (skills, experience, projects)
- Exclude noisy sections (education, certifications)
- Better semantic similarity computation
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re

logger = logging.getLogger(__name__)


def clean_semantic_text(text: str) -> str:
    """
    Clean text for semantic embedding
    Remove extra whitespace and noise
    """
    if not text:
        return ""
    text = text.lower()
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SemanticMatcher:
    """
    Fixed semantic matching using clean technical sections only
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading SentenceTransformer: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
    
    def _compute_cosine(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        text1_clean = clean_semantic_text(text1)
        text2_clean = clean_semantic_text(text2)
        
        if not text1_clean or not text2_clean:
            return 0.0
        
        try:
            embeddings = self.model.encode([text1_clean, text2_clean])
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Simple method to compute similarity between two texts.
        This is called from debug_pipeline and other components.
        """
        self._load_model()
        return self._compute_cosine(text1, text2)
    
    def get_sectioned_semantic_similarity(
        self,
        resume_sections: dict,
        job_role: str,
        requirements: str,
        job_description: str,
        extracted_jd_skills: list
    ) -> dict:
        """
        Compute semantic similarity using clean technical sections only
        
        Weighted as:
        - Skills section vs job requirements (40%)
        - Experience section vs job description (40%)
        - Projects section vs job role (20%)
        """
        self._load_model()
        
        # Extract technical sections only (exclude education, certifications)
        skills_text = clean_semantic_text(resume_sections.get("skills", ""))
        experience_text = clean_semantic_text(resume_sections.get("experience", ""))
        projects_text = clean_semantic_text(resume_sections.get("projects", ""))
        
        logger.info(f"Section lengths - Skills: {len(skills_text.split())}, "
                   f"Experience: {len(experience_text.split())}, "
                   f"Projects: {len(projects_text.split())}")
        
        # Build clean target texts (not full JD which is noisy)
        target_skills = clean_semantic_text(f"{requirements} {' '.join(extracted_jd_skills)}")
        if not target_skills:
            target_skills = clean_semantic_text(job_description[:500])  # Use JD summary
        
        target_experience = clean_semantic_text(f"{job_role} {job_description[:300]}")
        target_projects = clean_semantic_text(f"{job_role} responsibilities requirements")
        
        # Compute similarities
        sim_skills = self._compute_cosine(skills_text, target_skills) if skills_text else 0.5
        sim_experience = self._compute_cosine(experience_text, target_experience) if experience_text else 0.5
        sim_projects = self._compute_cosine(projects_text, target_projects) if projects_text else 0.4
        
        logger.info(f"Raw similarities - Skills: {sim_skills:.3f}, "
                   f"Experience: {sim_experience:.3f}, "
                   f"Projects: {sim_projects:.3f}")
        
        # Weights: Skills and Experience most important
        w_skills = 0.40
        w_experience = 0.40
        w_projects = 0.20
        
        # Compute final score
        semantic_score = (
            w_skills * sim_skills +
            w_experience * sim_experience +
            w_projects * sim_projects
        )
        semantic_score = max(0.0, min(1.0, semantic_score))
        
        logger.info(f"Final semantic score: {semantic_score:.4f}")
        
        return {
            "skills_similarity": round(sim_skills, 4),
            "projects_similarity": round(sim_projects, 4),
            "experience_similarity": round(sim_experience, 4),
            "semantic_similarity_score": round(semantic_score, 4)
        }


# Singleton
semantic_matcher = SemanticMatcher()


def get_sectioned_semantic_similarity(
    resume_sections: dict,
    job_role: str,
    requirements: str,
    job_description: str,
    extracted_jd_skills: list
) -> dict:
    """Convenience function"""
    return semantic_matcher.get_sectioned_semantic_similarity(
        resume_sections, job_role, requirements, job_description, extracted_jd_skills
    )
