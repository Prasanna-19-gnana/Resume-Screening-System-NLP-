"""
FEATURE ENGINEERING: Extract features from resume-JD pairs for ML model
Computes 7 key features:
1. semantic_similarity_score (0-1)
2. skills_exact_match_ratio (0-1)
3. skills_partial_match_ratio (0-1)
4. role_alignment_score (0-1)
5. number_of_matched_skills (count)
6. number_of_missing_skills (count)
7. top_sentence_similarity (0-1)
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract features from resume-JD pairs"""
    
    def __init__(self, skill_extractor, semantic_matcher, role_detector, context_matcher):
        """
        Initialize with required components
        
        Args:
            skill_extractor: SkillExtractor instance
            semantic_matcher: SemanticMatcher instance
            role_detector: RoleDetector instance
            context_matcher: ContextAwareMatcher instance
        """
        self.skill_extractor = skill_extractor
        self.semantic_matcher = semantic_matcher
        self.role_detector = role_detector
        self.context_matcher = context_matcher
    
    def extract_features(
        self,
        resume_text: str,
        job_description: str,
        requirements: str
    ) -> Dict[str, float]:
        """
        Extract all features for one resume-JD pair
        
        Returns:
        {
            "semantic_similarity_score": 0.65,
            "skills_exact_match_ratio": 0.80,
            "skills_partial_match_ratio": 0.15,
            "role_alignment_score": 0.50,
            "number_of_matched_skills": 5,
            "number_of_missing_skills": 3,
            "top_sentence_similarity": 0.75
        }
        """
        
        features = {}
        
        try:
            # Feature 1: Semantic Similarity Score
            # Simple similarity between resume and job description
            try:
                semantic_score = self.semantic_matcher.compute_similarity(
                    resume_text[:500], job_description[:500]
                )
                features["semantic_similarity_score"] = max(0.0, min(1.0, semantic_score))
            except Exception as e:
                logger.warning(f"Semantic similarity failed: {e}")
                features["semantic_similarity_score"] = 0.5
            
            # Feature 2, 3: Skill Matching (exact and partial)
            candidate_skills = self.skill_extractor.extract_skills(resume_text)
            required_skills = self.skill_extractor.extract_from_requirements(requirements)
            
            if not required_skills:
                required_skills = self.skill_extractor.extract_skills(job_description)
            
            # Simple exact match (case-insensitive)
            exact_matches = sum(
                1 for req in required_skills
                if any(req.lower() == cand.lower() for cand in candidate_skills)
            )
            exact_ratio = exact_matches / len(required_skills) if required_skills else 0.0
            features["skills_exact_match_ratio"] = max(0.0, min(1.0, exact_ratio))
            
            # Partial matches (substring matching)
            partial_matches = sum(
                1 for req in required_skills
                if any(req.lower() in cand.lower() or cand.lower() in req.lower()
                       for cand in candidate_skills)
                and req.lower() not in [c.lower() for c in candidate_skills]
            )
            partial_ratio = partial_matches / len(required_skills) if required_skills else 0.0
            features["skills_partial_match_ratio"] = max(0.0, min(1.0, partial_ratio))
            
            # Feature 4: Role Alignment Score
            try:
                detected_role = self.role_detector.detect_role(resume_text)
                # Extract job role from job description
                job_roles = ["ml engineer", "nlp engineer", "data scientist", "full stack", "backend"]
                jd_roles = [role for role in job_roles if role.lower() in job_description.lower()]
                
                if jd_roles and detected_role:
                    role_match = any(
                        jd_role.lower() in detected_role.lower() or detected_role.lower() in jd_role
                        for jd_role in jd_roles
                    )
                    features["role_alignment_score"] = 0.8 if role_match else 0.3
                else:
                    features["role_alignment_score"] = 0.5
            except Exception as e:
                logger.warning(f"Role detection failed: {e}")
                features["role_alignment_score"] = 0.5
            
            # Feature 5: Number of Matched Skills
            features["number_of_matched_skills"] = float(exact_matches)
            
            # Feature 6: Number of Missing Skills
            missing = len(required_skills) - exact_matches
            features["number_of_missing_skills"] = float(max(0, missing))
            
            # Feature 7: Top Sentence Similarity
            try:
                context_result = self.context_matcher.compute_sentence_similarities(
                    resume_text=resume_text,
                    job_description=job_description
                )
                
                max_sim = context_result.get("max_similarity", 0.0)
                features["top_sentence_similarity"] = max(0.0, min(1.0, max_sim))
            except Exception as e:
                logger.warning(f"Sentence similarity failed: {e}")
                features["top_sentence_similarity"] = 0.5
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return default features on error
            return self._get_default_features()
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values"""
        return {
            "semantic_similarity_score": 0.5,
            "skills_exact_match_ratio": 0.5,
            "skills_partial_match_ratio": 0.5,
            "role_alignment_score": 0.5,
            "number_of_matched_skills": 0.0,
            "number_of_missing_skills": 5.0,
            "top_sentence_similarity": 0.5
        }
    
    def extract_features_batch(
        self,
        resume_texts: List[str],
        job_description: str,
        requirements: str
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """
        Extract features for multiple resumes
        
        Returns:
            - Feature matrix (n_samples, 7)
            - List of feature dicts
        """
        feature_list = []
        features_matrix = []
        
        for resume_text in resume_texts:
            features = self.extract_features(
                resume_text=resume_text,
                job_description=job_description,
                requirements=requirements
            )
            feature_list.append(features)
            
            # Add to matrix in consistent order
            feature_vector = [
                features["semantic_similarity_score"],
                features["skills_exact_match_ratio"],
                features["skills_partial_match_ratio"],
                features["role_alignment_score"],
                features["number_of_matched_skills"],
                features["number_of_missing_skills"],
                features["top_sentence_similarity"]
            ]
            features_matrix.append(feature_vector)
        
        return np.array(features_matrix), feature_list
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Return feature names in order"""
        return [
            "semantic_similarity_score",
            "skills_exact_match_ratio",
            "skills_partial_match_ratio",
            "role_alignment_score",
            "number_of_matched_skills",
            "number_of_missing_skills",
            "top_sentence_similarity"
        ]
    
    @staticmethod
    def normalize_features(X: np.ndarray) -> np.ndarray:
        """
        Normalize features to 0-1 range (MinMax scaling)
        
        Handle different scales:
        - Ratios (0-1): already normalized
        - Counts: normalize by max value
        """
        X_normalized = X.copy().astype(float)
        
        # Columns 4, 5 are counts - normalize by max
        max_val = max(X[:, 4].max(), X[:, 5].max(), 1.0)
        X_normalized[:, 4] = X[:, 4] / max_val
        X_normalized[:, 5] = X[:, 5] / max_val
        
        return X_normalized
