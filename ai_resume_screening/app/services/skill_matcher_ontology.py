"""
ONTOLOGY-AWARE SKILL MATCHER
Matches skills with partial credit for related parent/child skills

Logic:
- Exact match → 1.0 points
- Candidate has parent, required is child → 0.7 points (good confidence)
- Candidate has child, required is parent → 0.5 points (partial match)
- No match → 0 points
"""

import logging
from typing import List, Dict, Tuple, Set

try:
    from .skill_ontology import (
        SKILL_ONTOLOGY,
        CHILD_TO_PARENTS,
        get_parents_of_skill,
        get_children_of_skill,
        is_parent_skill,
        is_child_skill
    )
except ImportError:
    from skill_ontology import (
        SKILL_ONTOLOGY,
        CHILD_TO_PARENTS,
        get_parents_of_skill,
        get_children_of_skill,
        is_parent_skill,
        is_child_skill
    )

logger = logging.getLogger(__name__)


class OntologyAwareSkillMatcher:
    """
    Matches skills using hierarchical ontology
    Provides partial credit for semantically related skills
    """
    
    def __init__(self):
        self.ontology = SKILL_ONTOLOGY
        self.child_to_parents = CHILD_TO_PARENTS
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize skill to lowercase"""
        return skill.lower().strip()
    
    def find_exact_match(self, candidate_skill: str, required_skill: str) -> bool:
        """Check for exact match (case-insensitive)"""
        cand_norm = self.normalize_skill(candidate_skill)
        req_norm = self.normalize_skill(required_skill)
        return cand_norm == req_norm
    
    def find_parent_match(self, candidate_skill: str, required_skill: str) -> float:
        """
        Candidate has parent skill, required is child.
        Example: candidate has "deep learning", required is "tensorflow"
        Return: 0.7 (good match, but not exact)
        """
        req_norm = self.normalize_skill(required_skill)
        cand_norm = self.normalize_skill(candidate_skill)
        
        # Get parents of the required skill
        parents = get_parents_of_skill(req_norm)
        
        # Check if candidate has any of those parents
        if cand_norm in parents:
            logger.debug(f"Parent match: {cand_norm} (parent) ← {req_norm} (child)")
            return 0.7  # 70% credit
        
        return 0.0
    
    def find_child_match(self, candidate_skill: str, required_skill: str) -> float:
        """
        Candidate has child skill, required is parent.
        Example: candidate has "pytorch", required is "deep learning"
        Return: 0.8 (proves they have the parent skill)
        """
        req_norm = self.normalize_skill(required_skill)
        cand_norm = self.normalize_skill(candidate_skill)
        
        # Get children of the required skill
        children = get_children_of_skill(req_norm)
        
        # Check if candidate has any of those children
        if cand_norm in children:
            logger.debug(f"Child match: {cand_norm} (child) → {req_norm} (parent)")
            return 0.8  # 80% credit (proves competency in parent skill)
        
        return 0.0
    
    def compute_match_score(self, candidate_skill: str, required_skill: str) -> Tuple[float, str]:
        """
        Compute match score between candidate and required skill
        Returns: (score 0.0-1.0, match_type)
        
        Match types:
        - "exact": Perfect match
        - "parent": Candidate has parent skill
        - "child": Candidate has child skill
        - "none": No match
        """
        # 1. Check exact match (highest priority)
        if self.find_exact_match(candidate_skill, required_skill):
            return 1.0, "exact"
        
        # 2. Check parent match (candidate stronger - 70%)
        parent_score = self.find_parent_match(candidate_skill, required_skill)
        if parent_score > 0:
            return parent_score, "parent"
        
        # 3. Check child match (candidate weaker but proves skill - 80%)
        child_score = self.find_child_match(candidate_skill, required_skill)
        if child_score > 0:
            return child_score, "child"
        
        # 4. No match
        return 0.0, "none"
    
    def match_skills(
        self,
        candidate_skills: List[str],
        required_skills: List[str]
    ) -> Dict[str, any]:
        """
        Match candidate skills against required skills
        Returns detailed breakdown of matches
        """
        cand_set = [self.normalize_skill(s) for s in candidate_skills]
        req_set = [self.normalize_skill(s) for s in required_skills]
        
        exact_matches = []
        partial_matches = []  # List of (required_skill, candidate_skill, score)
        missing_skills = []
        
        total_score = 0.0
        max_possible = len(req_set)
        
        for req_skill in req_set:
            best_score = 0.0
            best_cand = None
            best_type = "none"
            
            # Try to match with each candidate skill
            for cand_skill in cand_set:
                score, match_type = self.compute_match_score(cand_skill, req_skill)
                
                if score > best_score:
                    best_score = score
                    best_cand = cand_skill
                    best_type = match_type
            
            # Categorize the match
            if best_type == "exact":
                exact_matches.append(req_skill)
                total_score += 1.0
                logger.info(f"✅ EXACT: {req_skill}")
            elif best_type in ["parent", "child"]:
                partial_matches.append({
                    "required": req_skill,
                    "candidate": best_cand,
                    "score": best_score,
                    "match_type": best_type
                })
                total_score += best_score
                logger.info(f"🟡 PARTIAL ({best_type} match {best_score}): {req_skill} ← {best_cand}")
            else:
                missing_skills.append(req_skill)
                logger.info(f"❌ MISSING: {req_skill}")
        
        # Calculate final skill match score (0.0 - 1.0)
        skill_match_score = total_score / max_possible if max_possible > 0 else 1.0
        skill_match_score = max(0.0, min(1.0, skill_match_score))
        
        return {
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "missing_skills": missing_skills,
            "skill_match_score": skill_match_score,  # 0.0-1.0
            "score_breakdown": {
                "exact_count": len(exact_matches),
                "partial_count": len(partial_matches),
                "missing_count": len(missing_skills),
                "total_required": max_possible,
                "points_earned": total_score,
                "points_possible": max_possible
            }
        }


# Singleton
ontology_matcher = OntologyAwareSkillMatcher()


def match_skills(
    candidate_skills: List[str],
    required_skills: List[str]
) -> Dict[str, any]:
    """Convenience function"""
    return ontology_matcher.match_skills(candidate_skills, required_skills)
