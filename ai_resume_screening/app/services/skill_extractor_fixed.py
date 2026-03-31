"""
FIXED: Skill Extractor with Proper Normalization and Cleaning
- Clean text before tokenization
- Normalize skill aliases
- Correct missing skills detection
"""

import re
import logging
from typing import List, Tuple, Set, Dict

logger = logging.getLogger(__name__)

# Comprehensive skill normalization mapping
SKILL_ALIASES = {
    # Programming Languages
    "python": {"python", "py"},
    "java": {"java"},
    "javascript": {"javascript", "js"},
    "typescript": {"typescript", "ts"},
    "c++": {"c++", "cpp"},
    "c#": {"c#", "csharp", "c sharp"},
    "go": {"go", "golang"},
    "rust": {"rust"},
    "ruby": {"ruby"},
    "php": {"php"},
    
    # Machine Learning
    "machine learning": {"machine learning", "ml", "machine-learning"},
    "deep learning": {"deep learning", "dl", "deep-learning"},
    "nlp": {"nlp", "natural language processing", "natural-language-processing"},
    "computer vision": {"computer vision", "cv"},
    
    # ML Libraries
    "scikit-learn": {"scikit-learn", "sklearn", "scikit learn", "scikit-learn"},
    "tensorflow": {"tensorflow", "tf"},
    "pytorch": {"pytorch", "torch"},
    "keras": {"keras"},
    "xgboost": {"xgboost", "xgb"},
    "lightgbm": {"lightgbm", "lgbm"},
    
    # Data Processing
    "pandas": {"pandas", "pd"},
    "numpy": {"numpy", "np"},
    "scipy": {"scipy"},
    
    # Databases & Data Storage
    "sql": {"sql"},
    "postgresql": {"postgresql", "postgres", "postgre"},
    "mysql": {"mysql"},
    "mongodb": {"mongodb", "mongo"},
    "elasticsearch": {"elasticsearch", "elastic"},
    "redis": {"redis"},
    
    # Web Frameworks
    "react": {"react", "reactjs"},
    "vue": {"vue", "vuejs"},
    "angular": {"angular"},
    "node.js": {"node.js", "nodejs", "node"},
    "express": {"express", "expressjs"},
    "django": {"django"},
    "flask": {"flask"},
    "fastapi": {"fastapi"},
    
    # Cloud & DevOps
    "aws": {"aws", "amazon"},
    "azure": {"azure"},
    "gcp": {"gcp", "google cloud"},
    "docker": {"docker"},
    "kubernetes": {"kubernetes", "k8s"},
    "jenkins": {"jenkins"},
    "git": {"git", "github"},
    "ci/cd": {"ci/cd", "cicd", "continuous integration"},
    
    # Data Visualization
    "tableau": {"tableau"},
    "power bi": {"power bi", "powerbi", "power-bi"},
    "matplotlib": {"matplotlib"},
    "seaborn": {"seaborn"},
    "plotly": {"plotly"},
    
    # Other
    "agile": {"agile"},
    "scrum": {"scrum"},
    "linux": {"linux"},
    "html": {"html"},
    "css": {"css"},
    "rest api": {"rest api", "rest"},
    "graphql": {"graphql"},
    "spacy": {"spacy"},
}

# Build reverse mapping for faster lookup
NORMALIZED_SKILLS = {}
for canonical, aliases in SKILL_ALIASES.items():
    for alias in aliases:
        NORMALIZED_SKILLS[alias] = canonical


def clean_text(text: str) -> str:
    """
    Clean text by removing brackets, special chars, and extra whitespace
    """
    text = text.lower()
    # Remove content in brackets: (scikit-learn) → scikit-learn
    text = re.sub(r"\(([^)]*)\)", r" \1 ", text)
    text = re.sub(r"\[([^\]]*)\]", r" \1 ", text)
    text = re.sub(r"\{([^}]*)\}", r" \1 ", text)
    # Replace special chars with spaces (but keep hyphens for multi-word skills)
    text = re.sub(r"[^\w\s\-]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_skill(skill: str) -> str:
    """
    Normalize a skill to canonical form
    If skill not in mapping, return lowercase version
    """
    skill_clean = clean_text(skill).strip()
    return NORMALIZED_SKILLS.get(skill_clean, skill_clean)


def extract_skill_tokens(text: str) -> Set[str]:
    """
    Extract skill tokens from text
    Returns: set of normalized skill tokens
    """
    text_clean = clean_text(text)
    
    # Tokenize by spaces and hyphens
    tokens = re.split(r"[\s\-]+", text_clean)
    
    # Keep tokens that are 2+ chars and either:
    # - in our skill vocabulary, or
    # - look like tech terms
    valid_tokens = set()
    for token in tokens:
        if len(token) >= 2:
            normalized = normalize_skill(token)
            # Only keep if it's a known skill or looks technical
            if normalized in NORMALIZED_SKILLS.values() or any(
                c in token for c in ['+', '/', '#'] or len(token) > 3
            ):
                valid_tokens.add(normalized)
    
    return valid_tokens


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract all recognized skills from text
    Returns: list of unique normalized skills
    """
    text_clean = clean_text(text)
    found_skills = set()
    
    # Check for each skill using word boundaries
    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            # Escape special chars for regex
            escaped = re.escape(alias)
            # Look for word boundaries or hyphen boundaries
            pattern = rf"(?:^|\s|\-){escaped}(?:\s|\-|$)"
            if re.search(pattern, text_clean):
                found_skills.add(canonical)
                break  # Found this canonical skill, move to next
    
    return sorted(list(found_skills))


def match_skills(candidate_skills: List[str], required_skills: List[str]) -> Tuple[List[str], List[str]]:
    """
    Match candidate skills against required skills
    Returns: (matched_skills, missing_skills)
    """
    # Normalize both sets
    cand_set = set(normalize_skill(s) for s in candidate_skills)
    req_set = set(normalize_skill(s) for s in required_skills)
    
    matched = list(cand_set & req_set)
    missing = list(req_set - cand_set)
    
    return sorted(matched), sorted(missing)


class SkillExtractor:
    """
    Fixed skill extraction with proper normalization and cleaning
    """
    
    def __init__(self):
        self.skill_mapping = SKILL_ALIASES
        self.normalized_skills = NORMALIZED_SKILLS
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        return extract_skills_from_text(text)
    
    def extract_from_requirements(self, req_text: str) -> List[str]:
        """
        Extract skills from requirements section
        Handles both comma-separated and natural text
        """
        if not req_text.strip():
            return []
        
        # Try comma-separated first
        parts = re.split(r"[,;|]|\n", req_text)
        skills_from_csv = []
        for part in parts:
            part_clean = clean_text(part).strip()
            if part_clean and len(part_clean) > 2:
                # If it looks like a skill (has common tech chars or is long)
                if any(c in part_clean for c in ['+', '#', '.']) or len(part_clean) > 5:
                    normalized = normalize_skill(part_clean)
                    if normalized:
                        skills_from_csv.append(normalized)
        
        # Also extract using pattern matching
        skills_from_patterns = extract_skills_from_text(req_text)
        
        # Combine and deduplicate
        all_skills = list(set(skills_from_csv + skills_from_patterns))
        return sorted(all_skills)
    
    def evaluate_skills(
        self, 
        candidate_skills: List[str], 
        required_skills: List[str]
    ) -> Tuple[List[str], List[str], List[str], float]:
        """
        Evaluate skill match
        Returns: (matched, missing, extra, skill_score 0.0-1.0)
        """
        # Normalize
        cand_set = set(normalize_skill(s) for s in candidate_skills)
        req_set = set(normalize_skill(s) for s in required_skills)
        
        matched = list(cand_set & req_set)
        missing = list(req_set - cand_set)
        extra = list(cand_set - req_set)
        
        # Score: % of required skills matched
        if len(req_set) == 0:
            score = 1.0 if len(cand_set) > 0 else 0.0
        else:
            score = len(matched) / len(req_set)
        
        score = max(0.0, min(1.0, score))  # Clamp to 0-1
        
        logger.info(f"Skill Eval: {len(matched)} matched, {len(missing)} missing, score={score:.2f}")
        
        return sorted(matched), sorted(missing), sorted(extra), score


# Singleton
skill_extractor = SkillExtractor()
