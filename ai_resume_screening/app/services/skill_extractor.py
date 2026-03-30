import os
import json
import re
from typing import List, Tuple, Dict, Set
from .preprocessing import normalize_skill, clean_text

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ONTOLOGY_PATH = os.path.join(BASE_DIR, "data", "processed", "skill_ontology.json")

class SkillExtractor:
    def __init__(self):
        self.ontology = self._load_ontology()
        # Compile vocabulary from ontology keys and their children
        self.vocab = set(self.ontology.keys())
        for expanded in self.ontology.values():
            self.vocab.update(expanded)
            
    def _load_ontology(self) -> Dict[str, List[str]]:
        if os.path.exists(ONTOLOGY_PATH):
            with open(ONTOLOGY_PATH, 'r') as f:
                return json.load(f)
        else:
            print(f"[Warning] Ontology not found at {ONTOLOGY_PATH}. Falling back to default.")
            return {
                "machine learning": ["sklearn", "xgboost", "pandas", "numpy", "tensorflow", "pytorch"],
                "deep learning": ["tensorflow", "pytorch", "keras", "neural networks"],
                "nlp": ["transformers", "tokenization", "text classification", "chatbot", "spacy", "huggingface"],
                "web development": ["rest api", "frontend", "backend", "html", "css", "javascript"],
                "mern": ["mongodb", "express", "react", "nodejs"],
                "devops": ["docker", "kubernetes", "aws", "ci/cd", "jenkins", "terraform"]
            }

    def extract_skills(self, text: str) -> List[str]:
        """
        Extracts recognized skills out of unstructured text.
        In a robust system, this involves NER (spaCy). Here we map against our generated Vocab.
        """
        cleaned = clean_text(text)
        found_skills = set()
        
        # Word token sequence matching for multi-word and single-word skills
        for skill in self.vocab:
            # simple token boundary check
            # For exact phrase match, we use regex word boundaries
            escaped_skill = re.escape(skill)
            if re.search(rf"(?<!\w){escaped_skill}(?!\w)", cleaned):
                found_skills.add(normalize_skill(skill))
                
        return list(found_skills)

    def extract_from_requirements(self, req_text: str) -> List[str]:
        """
        Sometimes requirements are comma separated explicitly.
        """
        if not req_text.strip():
            return []
            
        parts = re.split(r'[,|;|\n|\u2022|\-]', req_text)
        extracted = []
        for p in parts:
            p_clean = p.strip()
            if p_clean and len(p_clean) > 1:
                extracted.append(normalize_skill(p_clean))
        
        # Merge with vocab detection
        detected = self.extract_skills(req_text)
        return list(set(extracted + detected))

    def evaluate_skills(self, candidate_skills: List[str], required_skills: List[str]) -> Tuple[List[str], List[str], List[str], float]:
        """
        Matches candidate explicit skills against requirements, applying Ontology Expansion (Layer 3).
        Returns: matched_skills, missing_skills, extra_skills, skill_match_score (0.0 - 1.0)
        """
        if not required_skills:
            return candidate_skills, [], [], 1.0
            
        req_set = set(required_skills)
        cand_set = set(candidate_skills)
        
        matched = set()
        missing = set()
        
        # Base credit
        total_possible = len(req_set)
        earned_score = 0.0
        
        for req in req_set:
            if req in cand_set:
                matched.add(req)
                earned_score += 1.0
            else:
                # Check ontology for partial credit
                # E.g. Require "tensorflow". Candidate has "deep learning".
                # Reverse lookup or forward lookup based on domain
                found_partial = False
                
                # If the candidate has a parent skill that covers the requirement
                for parent, children in self.ontology.items():
                    if req in children and parent in cand_set:
                        matched.add(req)  # Add to matched but via expansion
                        earned_score += 0.5 # Partial credit!
                        found_partial = True
                        break
                        
                # If candidate has a specific child tool that proves the parent requirement
                if not found_partial and req in self.ontology:
                    for child in self.ontology[req]:
                        if child in cand_set:
                            matched.add(req)
                            earned_score += 0.8 # Specific proves the parent heavily
                            found_partial = True
                            break
                            
                if not found_partial:
                    missing.add(req)
                    
        extra = list(cand_set - matched)
        
        skill_score = earned_score / total_possible if total_possible > 0 else 0.0
        # normalize clamp
        skill_score = max(0.0, min(1.0, skill_score))
        
        return list(matched), list(missing), extra, skill_score
