"""
UPGRADED: Scoring Engine with Ontology-Aware Skill Matching
- Exact matches: full credit
- Related skills: partial credit
- Better fairness for similar but not identical skills
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class UpgradedAIScorer:
    """
    Upgraded AI scorer using ontology-aware skill matching
    """
    
    def __init__(self, skill_extractor, semantic_matcher, role_detector, ontology_matcher):
        self.skill_extractor = skill_extractor
        self.semantic_matcher = semantic_matcher
        self.role_detector = role_detector
        self.ontology_matcher = ontology_matcher
    
    def score_candidate(
        self,
        parsed_resume: dict,
        job_role: str,
        requirements: str,
        job_description: str,
        resume_semantic_text: str
    ) -> Dict[str, Any]:
        """
        Score a candidate using ontology-aware matching
        
        Scoring formula (UPGRADED):
        - Semantic Similarity: 35%  (content relevance)
        - Skill Matching (with ontology): 45%  (INCREASED - most important)
        - Role Alignment: 20%  (role match)
        """
        
        logger.info(f"Scoring candidate for role: {job_role}")
        
        # ====== 1. SKILL MATCHING WITH ONTOLOGY (45%) ======
        candidate_skills = self.skill_extractor.extract_skills(resume_semantic_text)
        required_skills = self.skill_extractor.extract_from_requirements(requirements)
        if not required_skills:
            required_skills = self.skill_extractor.extract_skills(job_description)
        
        # Use ontology-aware matching
        match_result = self.ontology_matcher.match_skills(
            candidate_skills,
            required_skills
        )
        
        skill_score = match_result["skill_match_score"]
        exact_matches = match_result["exact_matches"]
        partial_matches = match_result["partial_matches"]
        missing_skills = match_result["missing_skills"]
        
        logger.info(f"Skills: {len(exact_matches)} exact, {len(partial_matches)} partial, {len(missing_skills)} missing")
        logger.info(f"Skill score: {skill_score:.3f}")
        
        # ====== 2. SEMANTIC SIMILARITY (35%) ======
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
        
        # ====== 4. FINAL SCORING (UPGRADED WEIGHTS) ======
        # Increased skill weight from 40% to 45%
        # Decreased semantic from 40% to 35% (but still important)
        # Kept role at 20%
        final_score_normalized = (
            0.35 * semantic_score +
            0.45 * skill_score +
            0.20 * role_alignment_score
        )
        
        # Scale to 0-100
        final_score = final_score_normalized * 100
        final_score = max(0.0, min(100.0, final_score))
        
        logger.info(f"Final score: {final_score:.1f}/100")
        
        # ====== 5. RECOMMENDATION TIER ======
        if final_score >= 80:
            recommendation = "🌟 STRONG MATCH - Highly recommended for interview"
            confidence = "High"
        elif final_score >= 70:
            recommendation = "✅ GOOD MATCH - Recommended for interview"
            confidence = "Medium-High"
        elif final_score >= 55:
            recommendation = "⚡ MODERATE MATCH - Consider for interview"
            confidence = "Medium"
        elif final_score >= 40:
            recommendation = "⚠️ WEAK MATCH - May need screening"
            confidence = "Low"
        else:
            recommendation = "❌ POOR MATCH - Not recommended"
            confidence = "Very Low"
        
        return {
            # Main scores (0-100)
            "final_score": round(final_score, 1),
            "semantic_similarity_score": round(semantic_score * 100, 1),
            "skill_match_score": round(skill_score * 100, 1),
            "role_alignment_score": round(role_alignment_score * 100, 1),
            
            # Skill matching details (UPGRADED)
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "missing_skills": missing_skills,
            "skill_breakdown": match_result["score_breakdown"],
            
            # Role details
            "detected_role": detected_role,
            
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


def create_upgraded_scorer(skill_extractor, semantic_matcher, role_detector, ontology_matcher):
    """Factory function to create upgraded scorer"""
    return UpgradedAIScorer(skill_extractor, semantic_matcher, role_detector, ontology_matcher)
