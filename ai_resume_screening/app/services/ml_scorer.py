"""
ML INFERENCE: Use trained model for resume scoring

This replaces the manual rule-based scoring with ML predictions
"""

import logging
from typing import Dict, Optional, Any
import numpy as np
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class MLScorer:
    """Use trained ML model to score resumes"""
    
    def __init__(
        self,
        feature_engineer,
        model_path: Optional[str] = None,
        fallback_scorer = None
    ):
        """
        Initialize ML scorer
        
        Args:
            feature_engineer: FeatureEngineer instance
            model_path: Path to trained model (optional, can be set later)
            fallback_scorer: Scorer to use if ML model fails (optional)
        """
        self.feature_engineer = feature_engineer
        self.fallback_scorer = fallback_scorer
        self.model = None
        self.feature_names = feature_engineer.get_feature_names()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load trained model from disk"""
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"ML model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            self.model = None
    
    def score_resume(
        self,
        resume_text: str,
        job_description: str,
        requirements: str,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Score a resume using ML model
        
        Returns:
        {
            "score": 72.5,
            "ML_model": "RandomForest",
            "confidence": 0.85,
            "features": {...} (optional)
        }
        """
        
        result = {
            "score": 50.0,  # Default
            "ML_model": None,
            "confidence": 0.0,
            "method": "fallback"
        }
        
        # If no ML model, use fallback
        if self.model is None:
            if self.fallback_scorer:
                try:
                    fb_result = self.fallback_scorer.score_resume(
                        resume_text=resume_text,
                        job_description=job_description,
                        requirements=requirements
                    )
                    result["score"] = fb_result.get("score", 50.0)
                    result["method"] = "fallback_scorer"
                    logger.info(f"Using fallback scorer, score={result['score']:.1f}")
                except Exception as e:
                    logger.error(f"Fallback scorer failed: {e}")
            else:
                logger.warning("No ML model and no fallback scorer")
            
            return result
        
        try:
            # Extract features
            features = self.feature_engineer.extract_features(
                resume_text=resume_text,
                job_description=job_description,
                requirements=requirements
            )
            
            # Convert to feature vector
            feature_vector = np.array([
                features["semantic_similarity_score"],
                features["skills_exact_match_ratio"],
                features["skills_partial_match_ratio"],
                features["role_alignment_score"],
                features["number_of_matched_skills"],
                features["number_of_missing_skills"],
                features["top_sentence_similarity"]
            ]).reshape(1, -1)
            
            # Make prediction
            predicted_score_normalized = self.model.predict(feature_vector)[0]
            
            # Convert to 0-100 scale
            predicted_score = float(predicted_score_normalized * 100)
            predicted_score = max(0.0, min(100.0, predicted_score))  # Clamp to valid range
            
            # Estimate confidence based on feature consistency
            feature_values = list(features.values())
            feature_std = np.std(feature_values[:7])  # First 7 are normalized
            confidence = 1.0 - (feature_std * 0.1)  # Simple heuristic
            confidence = max(0.0, min(1.0, confidence))
            
            result.update({
                "score": predicted_score,
                "ML_model": self._get_model_name(),
                "confidence": confidence,
                "method": "ml_model"
            })
            
            if return_features:
                result["features"] = features
            
            logger.info(
                f"ML prediction: score={predicted_score:.1f}, "
                f"model={self._get_model_name()}, confidence={confidence:.2f}"
            )
        
        except Exception as e:
            logger.error(f"ML scoring failed: {e}")
            
            # Fallback to rule-based if available
            if self.fallback_scorer:
                try:
                    fb_result = self.fallback_scorer.score_resume(
                        resume_text=resume_text,
                        job_description=job_description,
                        requirements=requirements
                    )
                    result["score"] = fb_result.get("score", 50.0)
                    result["method"] = "fallback_after_ml_error"
                except:
                    pass
        
        return result
    
    def score_batch(
        self,
        resume_texts: list,
        job_description: str,
        requirements: str,
        return_features: bool = False
    ) -> list:
        """
        Score multiple resumes
        
        Args:
            resume_texts: List of resume texts
            job_description: Job description
            requirements: Required skills/qualifications
            return_features: Include feature vectors in results
        
        Returns:
            List of scoring results
        """
        
        results = []
        
        for idx, resume_text in enumerate(resume_texts):
            try:
                result = self.score_resume(
                    resume_text=resume_text,
                    job_description=job_description,
                    requirements=requirements,
                    return_features=return_features
                )
                result["resume_idx"] = idx
                results.append(result)
            
            except Exception as e:
                logger.error(f"Error scoring resume {idx}: {e}")
                results.append({
                    "resume_idx": idx,
                    "score": 50.0,
                    "error": str(e)
                })
        
        return results
    
    def _get_model_name(self) -> str:
        """Identify which ML model is loaded"""
        
        if self.model is None:
            return None
        
        model_class_name = self.model.__class__.__name__
        
        if 'RandomForest' in model_class_name:
            return 'RandomForest'
        elif 'XGB' in model_class_name or 'xgb' in model_class_name:
            return 'XGBoost'
        else:
            return model_class_name
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        
        if self.model is None:
            return {
                "status": "no_model",
                "model_name": None
            }
        
        info = {
            "status": "loaded",
            "model_name": self._get_model_name(),
            "model_class": self.model.__class__.__name__
        }
        
        # Add model-specific info
        if hasattr(self.model, 'n_estimators'):
            info["n_estimators"] = int(self.model.n_estimators)
        
        if hasattr(self.model, 'max_depth'):
            info["max_depth"] = self.model.max_depth
        
        if hasattr(self.model, 'feature_importances_'):
            info["feature_importances"] = dict(
                zip(self.feature_names, self.model.feature_importances_)
            )
        
        return info


class EnsembleScorer:
    """Ensemble scorer: combine ML and rule-based scoring"""
    
    def __init__(self, ml_scorer, rule_scorer, ml_weight: float = 0.7):
        """
        Combine ML and rule-based scores
        
        Args:
            ml_scorer: MLScorer instance
            rule_scorer: Current/fallback scorer instance
            ml_weight: Weight for ML score (0-1)
        """
        self.ml_scorer = ml_scorer
        self.rule_scorer = rule_scorer
        self.ml_weight = ml_weight
        self.rule_weight = 1.0 - ml_weight
    
    def score_resume(
        self,
        resume_text: str,
        job_description: str,
        requirements: str,
        return_components: bool = False
    ) -> Dict[str, Any]:
        """
        Score using ensemble of ML + rule-based
        
        Returns:
        {
            "score": 65.0,
            "ml_score": 72.0,
            "rule_score": 55.0,
            "method": "ensemble"
        }
        """
        
        # Get ML score
        ml_result = self.ml_scorer.score_resume(
            resume_text=resume_text,
            job_description=job_description,
            requirements=requirements
        )
        ml_score = ml_result.get("score", 50.0)
        
        # Get rule-based score
        rule_result = self.rule_scorer.score_resume(
            resume_text=resume_text,
            job_description=job_description,
            requirements=requirements
        )
        rule_score = rule_result.get("score", 50.0)
        
        # Ensemble prediction
        ensemble_score = (
            ml_score * self.ml_weight +
            rule_score * self.rule_weight
        )
        
        result = {
            "score": ensemble_score,
            "method": "ensemble",
            "ml_weight": self.ml_weight,
            "rule_weight": self.rule_weight
        }
        
        if return_components:
            result["ml_score"] = ml_score
            result["rule_score"] = rule_score
        
        return result
