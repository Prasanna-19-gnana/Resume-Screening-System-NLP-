import os
import joblib
import logging
import numpy as np
from typing import Optional, Dict
from .preprocessing import clean_text

logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "role_classifier.pkl")

class RoleClassifier:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.loaded = False
        self._load_models()

    def _load_models(self):
        if os.path.exists(VECTORIZER_PATH) and os.path.exists(CLASSIFIER_PATH):
            try:
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                self.classifier = joblib.load(CLASSIFIER_PATH)
                self.loaded = True
                logger.info("Loaded Role Classifier models successfully.")
            except Exception as e:
                logger.error(f"Error loading Role Classifier models: {e}")
        else:
            logger.warning("Role Classifier models not found. Run training script.")

    def predict_resume_role_probabilities(self, resume_semantic_text: str) -> dict:
        """ Returns {"RoleA": prob_float, "RoleB": prob_float} """
        if not resume_semantic_text.strip():
            return {"Unknown": 1.0}
            
        if not self.loaded:
            logger.warning("Inference fallback used. Models not loaded.")
            return {"Unknown": 1.0}
            
        cleaned = clean_text(resume_semantic_text)
        vec = self.vectorizer.transform([cleaned])
        
        # Logistic Regression Probability
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(vec)[0]
            classes = self.classifier.classes_
            class_probs = {classes[i]: probs[i] for i in range(len(classes))}
            sorted_probs = dict(sorted(class_probs.items(), key=lambda item: item[1], reverse=True)[:3])
            return sorted_probs
            
        else: 
            # Linear SVM specific conversion to probabilities
            decision = self.classifier.decision_function(vec)[0]
            classes = self.classifier.classes_
            exp_d = np.exp(decision - np.max(decision))
            probs = exp_d / exp_d.sum()
            class_probs = {classes[i]: probs[i] for i in range(len(classes))}
            sorted_probs = dict(sorted(class_probs.items(), key=lambda item: item[1], reverse=True)[:3])
            return sorted_probs

    def calculate_role_alignment(self, probabilities_dict: dict, target_job_role: str) -> float:
        """ Multi-label alignment! Extracts exact probability mapping safely. """
        r2 = target_job_role.lower()
        best_match_prob = 0.0
        
        for cand_role, prob in probabilities_dict.items():
            r1 = cand_role.lower()
            if r1 == r2 or r1 in r2 or r2 in r1:
                return prob  # Perfect mathematical alignment metric!
                
            # Domain clusters fallback
            data_aliases = ["data scientist", "data analyst", "machine learning", "nlp engineer"]
            dev_aliases = ["software engineer", "developer", "full stack", "backend", "mern"]
            
            if any(a in r1 for a in data_aliases) and any(a in r2 for a in data_aliases):
                best_match_prob = max(best_match_prob, prob * 0.8) # Penalize slight drift
                
            if any(a in r1 for a in dev_aliases) and any(a in r2 for a in dev_aliases):
                best_match_prob = max(best_match_prob, prob * 0.8)
                
        return max(0.1, best_match_prob)

# Singleton
role_classifier_svc = RoleClassifier()

def predict_role_probabilities(resume_semantic_text: str) -> dict:
    return role_classifier_svc.predict_resume_role_probabilities(resume_semantic_text)
