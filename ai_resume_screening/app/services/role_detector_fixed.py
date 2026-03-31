"""
NEW: Role Detector with Rule-Based Fallback
Detects job role from resume content with sensible defaults
"""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class RoleDetector:
    """
    Rule-based role detection from resume text
    Fallback when ML model is unavailable
    """
    
    # Role detection rules (patterns → roles)
    ROLE_RULES = {
        "Data Scientist": [
            r"machine learning|ml|deep learning|neural network|model|sklearn|tensorflow|pytorch|statistics|statistical",
            r"nlp|natural language|computer vision|image processing",
            r"data science|analytics|predictive|classification|regression"
        ],
        "Data Engineer": [
            r"data pipeline|etl|spark|hadoop|kafka|dataflow",
            r"data warehouse|big data|distributed|sql|dbt",
            r"data infrastructure|lake|parquet|airflow"
        ],
        "Full Stack Developer": [
            r"react|vue|angular|frontend|backend|node\.js|express|django|flask",
            r"full stack|mern|mean|rest api|graphql",
            r"web.*development|responsive.*design"
        ],
        "Frontend Developer": [
            r"react|vue|angular|javascript|typescript|html|css|ui|ux",
            r"frontend|client.*side|responsive|accessibility",
            r"web design|figma|adobe"
        ],
        "Backend Developer": [
            r"node\.js|express|django|flask|fastapi|spring|java",
            r"database|sql|api|backend|server.*side",
            r"microservice|kubernetes|docker|devops"
        ],
        "DevOps Engineer": [
            r"docker|kubernetes|jenkins|ci/cd|terraform|ansible",
            r"devops|infrastructure|deployment|cloud|aws|azure|gcp",
            r"monitoring|logging|prometheus|grafana"
        ],
        "Cloud Architect": [
            r"aws|azure|gcp|cloud|infrastructure|architecture",
            r"terraform|cloudformation|deployment|scaling",
            r"migration|hybrid|serverless"
        ],
        "Machine Learning Engineer": [
            r"machine learning|ml|deep learning|neural network|pytorch|tensorflow",
            r"training|optimization|model|algorithm|feature.*engineering",
            r"mlops|model.*serving|inference"
        ],
        "Software Engineer": [
            r"software.*engineering|programming|development|coding",
            r"algorithms|design.*pattern|solid.*principles",
            r"testing|debugging|git"
        ]
    }
    
    # Default fallback
    DEFAULT_ROLE = "Software Engineer"
    
    def __init__(self):
        # Compile patterns for efficiency
        self.compiled_rules = {}
        for role, patterns in self.ROLE_RULES.items():
            self.compiled_rules[role] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def detect_role(self, text: str, detected_role: str = None) -> str:
        """
        Detect job role from resume text
        
        Args:
            text: Resume/profile text
            detected_role: Pre-detected role (from ML model), used as tiebreaker
        
        Returns:
            Detected role name
        """
        if not text or not text.strip():
            logger.warning("No text provided for role detection")
            return self.DEFAULT_ROLE
        
        text_lower = text.lower()
        role_scores = {}
        
        # Score each role based on pattern matches
        for role, patterns in self.compiled_rules.items():
            score = 0
            for pattern in patterns:
                if pattern.search(text_lower):
                    score += 1
            role_scores[role] = score
        
        # Find role with highest score
        if max(role_scores.values()) > 0:
            best_role = max(role_scores, key=role_scores.get)
            logger.info(f"Detected role: {best_role} (score={role_scores[best_role]})")
            return best_role
        else:
            # No patterns matched, use fallback
            logger.info("No role patterns matched, using default")
            return detected_role or self.DEFAULT_ROLE
    
    def get_role_alignment_score(self, detected_role: str, job_role: str) -> float:
        """
        Score role alignment (0.0-1.0)
        
        1.0 = perfect match
        0.8 = similar role
        0.5 = different but related
        0.3 = very different
        """
        if not job_role or not detected_role:
            return 0.5
        
        job_lower = job_role.lower()
        detected_lower = detected_role.lower()
        
        # Exact match
        if job_lower == detected_lower:
            return 1.0
        
        # Partial match (e.g., "Engineer" in both)
        job_words = set(job_lower.split())
        detected_words = set(detected_lower.split())
        overlap = job_words & detected_words
        
        if overlap and len(overlap) >= 2:
            return 0.8
        elif overlap:
            return 0.6
        
        # Similar domains (hardcoded mapping)
        similar_domains = {
            "data scientist": {"data engineer", "ml engineer", "machine learning engineer"},
            "frontend developer": {"full stack developer"},
            "backend developer": {"full stack developer"},
            "devops engineer": {"cloud architect", "infrastructure engineer"},
        }
        
        for role_a, role_b_set in similar_domains.items():
            if (detected_lower == role_a and job_lower in role_b_set) or \
               (job_lower == role_a and detected_lower in role_b_set):
                return 0.7
        
        # Different roles
        return 0.4


# Singleton
role_detector = RoleDetector()


def detect_role(text: str, detected_role: str = None) -> str:
    """Convenience function"""
    return role_detector.detect_role(text, detected_role)


def get_role_alignment_score(detected_role: str, job_role: str) -> float:
    """Convenience function"""
    return role_detector.get_role_alignment_score(detected_role, job_role)
