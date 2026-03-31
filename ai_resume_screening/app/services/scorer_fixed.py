"""
FIXED: Scoring Engine with Correct Weights and Logic
- Balanced scoring: 40% semantic + 40% skills + 20% role
- No contradictions
- Proper score distribution (0-100)
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class AIScorer:
    """
    Fixed AI scoring with balanced weights
    """
    
    def __init__(self, skill_extractor, semantic_matcher, role_detector):
        self.skill_extractor = skill_extractor
        self.semantic_matcher = semantic_matcher
        self.role_detector = role_detector
    
    def score_candidate(
        self,
        parsed_resume: dict,
        job_role: str,
        requirements: str,
        job_description: str,
        resume_semantic_text: str
    ) -> Dict[str, Any]:
        """
        Score a candidate with fixed balanced weights
        
        Scoring formula:
        - Semantic Similarity: 40%  (resume content matches job)
        - Skill Matching: 40%       (required skills present)
        - Role Alignment: 20%       (resume role matches job role)
        
        Final Score: 0-100
        """
        
        logger.info(f"Scoring candidate for role: {job_role}")
        
        # ====== 1. SKILL MATCHING (40%) ======
        candidate_skills = self.skill_extractor.extract_skills(resume_semantic_text)
        required_skills = self.skill_extractor.extract_from_requirements(requirements)
        if not required_skills:  # If no explicit requirements, extract from JD
            required_skills = self.skill_extractor.extract_skills(job_description)
        
        matched_skills, missing_skills, extra_skills, skill_score = self.skill_extractor.evaluate_skills(
            candidate_skills,
            required_skills
        )
        
        logger.info(f"Skills: {len(matched_skills)} matched, {len(missing_skills)} missing")
        logger.info(f"Skill score: {skill_score:.3f}")
        
        # ====== 2. SEMANTIC SIMILARITY (40%) ======
        sem_result = self.semantic_matcher.get_sectioned_semantic_similarity(
            resume_sections=parsed_resume,
            job_role=job_role,
            requirements=requirements,
            job_description=job_description,
            extracted_jd_skills=required_skills
        )
        semantic_score = sem_result["semantic_similarity_score"]
        
        logger.info(f"Semantic score: {semantic_score:.3f}")
        
        # ====== 3. ROLE ALIGNMENT (20%) ======
        detected_role = self.role_detector.detect_role(resume_semantic_text)
        role_alignment_score = self.role_detector.get_role_alignment_score(detected_role, job_role)
        
        logger.info(f"Detected role: {detected_role}")
        logger.info(f"Role alignment score: {role_alignment_score:.3f}")
        
        # ====== 4. FINAL SCORING (BALANCED FORMULA) ======
        # Equal weight to semantic and skills (most important)
        # Lower weight to role (less critical)
        final_score_normalized = (
            0.40 * semantic_score +
            0.40 * skill_score +
            0.20 * role_alignment_score
        )
        
        # Scale to 0-100
        final_score = final_score_normalized * 100
        final_score = max(0.0, min(100.0, final_score))  # Clamp to 0-100
        
        logger.info(f"Final score: {final_score:.1f}/100")
        
        # ====== 5. RECOMMENDATION TIER ======
        if final_score >= 80:
            recommendation = "🌟 STRONG MATCH - Highly recommended for interview"
            confidence = "High"
        elif final_score >= 65:
            recommendation = "✅ GOOD MATCH - Recommended for interview"
            confidence = "Medium-High"
        elif final_score >= 50:
            recommendation = "⚡ MODERATE MATCH - Consider for interview"
            confidence = "Medium"
        elif final_score >= 35:
            recommendation = "⚠️ WEAK MATCH - May need screening"
            confidence = "Low"
        else:
            recommendation = "❌ POOR MATCH - Not recommended"
            confidence = "Very Low"
        
        return {
            # Scores (0-100)
            "final_score": round(final_score, 1),
            "semantic_similarity_score": round(semantic_score * 100, 1),
            "skill_match_score": round(skill_score * 100, 1),
            "role_alignment_score": round(role_alignment_score * 100, 1),
            
            # Details
            "detected_role": detected_role,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "extra_skills": extra_skills,
            
            # Semantic details
            "skills_similarity": sem_result["skills_similarity"] * 100,
            "experience_similarity": sem_result["experience_similarity"] * 100,
            "projects_similarity": sem_result["projects_similarity"] * 100,
            
            # Recommendation
            "recommendation": recommendation,
            "confidence": confidence,
            
            # Metadata
            "required_skills": required_skills,
            "candidate_skills": candidate_skills,
        }


def create_scorer(skill_extractor, semantic_matcher, role_detector):
    """Factory function to create scorer with dependencies"""
    return AIScorer(skill_extractor, semantic_matcher, role_detector)
