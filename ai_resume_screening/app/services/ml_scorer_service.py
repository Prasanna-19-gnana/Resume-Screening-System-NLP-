"""
ML SCORER SERVICE: Production-ready ML-based scoring for API

This module provides a drop-in replacement for AIScorer that uses the trained ML model.
Can operate in 3 modes:
1. ML-only: Pure ML predictions
2. Hybrid: ML + rule-based (weighted blend)
3. Fallback: Uses rule-based if ML unavailable
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from .feature_engineering import FeatureEngineer
from .ml_scorer import MLScorer, EnsembleScorer
from .skill_extractor_fixed import SkillExtractor
from .semantic_matcher_fixed import SemanticMatcher
from .role_detector_fixed import RoleDetector
from .context_matcher import ContextAwareMatcher
from .scorer import AIScorer

logger = logging.getLogger(__name__)


class MLScorerService:
    """Production ML scorer service with fallback support"""
    
    def __init__(
        self,
        mode: str = "hybrid",
        model_path: Optional[str] = None,
        ml_weight: float = 0.7
    ):
        """
        Initialize ML scorer service
        
        Args:
            mode: "ml", "hybrid", or "fallback"
            model_path: Path to trained model (auto-detected if None)
            ml_weight: Weight for ML score in hybrid mode (0-1)
        """
        
        self.mode = mode
        self.ml_weight = ml_weight
        self.rule_weight = 1.0 - ml_weight
        
        # Initialize components
        try:
            self.skill_extractor = SkillExtractor()
            self.semantic_matcher = SemanticMatcher()
            self.role_detector = RoleDetector()
            self.context_matcher = ContextAwareMatcher()
            
            self.feature_engineer = FeatureEngineer(
                self.skill_extractor,
                self.semantic_matcher,
                self.role_detector,
                self.context_matcher
            )
            
            self.rule_scorer = AIScorer()
            logger.info("✅ Components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            self.feature_engineer = None
            self.rule_scorer = None
        
        # Initialize ML model
        self.ml_scorer = None
        self._load_ml_model(model_path)
    
    def _load_ml_model(self, model_path: Optional[str]) -> None:
        """Load ML model from disk"""
        
        if model_path is None:
            # Try auto-detection
            default_path = Path(__file__).parent.parent.parent / "models" / "ml_scorer_rf.pkl"
            if default_path.exists():
                model_path = str(default_path)
            else:
                logger.warning(f"ML model not found at {default_path}")
                return
        
        try:
            self.ml_scorer = MLScorer(
                feature_engineer=self.feature_engineer,
                model_path=model_path,
                fallback_scorer=self.rule_scorer
            )
            logger.info(f"✅ ML model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.ml_scorer = None
    
    def score_candidate(
        self,
        parsed_resume_dict: dict,
        job_role: str,
        requirements: str,
        job_description: str,
        resume_semantic_text: str
    ) -> Dict[str, Any]:
        """
        Score a candidate (API-compatible interface)
        
        Returns:
        {
            "final_score": 0.75,
            "predicted_resume_role": "ML Engineer",
            "role_alignment_score": 0.85,
            "semantic_similarity_score": 0.72,
            "skill_match_score": 0.68,
            "matched_skills": ["Python", "ML"],
            "missing_skills": ["Spark"],
            "recommendation": "Moderate Match",
            "ml_score": 0.75,
            "rule_score": 0.70,
            "scoring_method": "ml" | "hybrid" | "fallback",
            ...
        }
        """
        
        logger.info(f"Scoring candidate for role: {job_role}")
        
        # Get rule-based score (always for comparison)
        rule_result = self.rule_scorer.score_candidate(
            parsed_resume_dict=parsed_resume_dict,
            job_role=job_role,
            requirements=requirements,
            job_description=job_description,
            resume_semantic_text=resume_semantic_text
        )
        
        rule_score = rule_result.get("final_score", 0.5)
        
        # Select scoring method based on mode
        if self.mode == "fallback":
            final_result = rule_result
            final_result["scoring_method"] = "rule_based"
            return final_result
        
        # Try ML scoring
        if self.ml_scorer and self.feature_engineer:
            try:
                ml_result = self.ml_scorer.score_resume(
                    resume_text=resume_semantic_text,
                    job_description=job_description,
                    requirements=requirements,
                    return_features=False
                )
                
                ml_score = ml_result.get("score", 50) / 100.0  # Convert 0-100 to 0-1
                
                if self.mode == "ml":
                    # Pure ML scoring
                    final_score = ml_score
                    method = "ml_only"
                else:  # hybrid
                    # Blend ML + rule-based
                    final_score = (ml_score * self.ml_weight) + (rule_score * self.rule_weight)
                    method = "hybrid"
                
                # Return hybrid result with both scores
                result = rule_result.copy()
                result["final_score"] = max(0.0, min(1.0, final_score))
                result["ml_score"] = ml_score
                result["rule_score"] = rule_score
                result["ml_weight"] = self.ml_weight
                result["rule_weight"] = self.rule_weight
                result["scoring_method"] = method
                result["recommendation"] = self._get_recommendation(result["final_score"])
                
                logger.info(
                    f"Scoring complete: ML={ml_score:.3f}, "
                    f"Rule={rule_score:.3f}, Final={result['final_score']:.3f}, "
                    f"Method={method}"
                )
                
                return result
            
            except Exception as e:
                logger.error(f"ML scoring failed, falling back to rule-based: {e}")
                rule_result["scoring_method"] = "fallback_after_error"
                return rule_result
        
        # ML not available, use rule-based
        logger.warning("ML scorer not available, using rule-based scoring")
        rule_result["scoring_method"] = "rule_based_fallback"
        return rule_result
    
    def score_resume(
        self,
        resume_text: str,
        job_description: str,
        requirements: str
    ) -> Dict[str, Any]:
        """
        Simple score_resume interface for components that don't have parsed resume dict.
        Converts simple text inputs into the format needed by score_candidate.
        
        Args:
            resume_text: Raw resume text
            job_description: Raw job description text
            requirements: Requirements string
        
        Returns:
            Dict with score and metadata
        """
        
        # Convert resume text to minimal parsed dict format (score_candidate expects this)
        parsed_resume_dict = {
            "summary": resume_text[:500],
            "skills": resume_text[:200],
            "experience": resume_text,
            "education": "",
            "projects": "",
            "full_text": resume_text
        }
        
        # Detect job role from requirement or default
        job_role = requirements.split(",")[0].strip() if requirements else "General"
        
        # Call main scoring method
        result = self.score_candidate(
            parsed_resume_dict=parsed_resume_dict,
            job_role=job_role,
            requirements=requirements,
            job_description=job_description,
            resume_semantic_text=resume_text
        )
        
        # Ensure result has 'score' key for compatibility
        if "final_score" in result and "score" not in result:
            result["score"] = result["final_score"] * 100  # Convert 0-1 to 0-100
        
        return result
    
    @staticmethod
    def _get_recommendation(score: float) -> str:
        """Get recommendation label from score"""
        if score >= 0.80:
            return "Strong Match"
        elif score >= 0.65:
            return "Moderate Match"
        elif score >= 0.45:
            return "Weak Match"
        else:
            return "Poor Match"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ML model"""
        
        info = {
            "mode": self.mode,
            "ml_weight": self.ml_weight,
            "rule_weight": self.rule_weight,
            "ml_model_available": self.ml_scorer is not None and self.ml_scorer.model is not None
        }
        
        if self.ml_scorer and self.ml_scorer.model:
            info["ml_model_info"] = self.ml_scorer.get_model_info()
        
        return info


# Create singleton service instances
# Default: hybrid mode (70% ML, 30% rule-based)
ml_scorer_service = MLScorerService(mode="hybrid", ml_weight=0.7)

# Also provide direct access to rule-based scorer
from .scorer import scorer_service as rule_scorer_service
