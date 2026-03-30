import re

# Extended dictionary of common tech and business skills for realistic matching
SKILL_DB = {
    # Engineering & Tech
    "python", "java", "c++", "c#", "javascript", "typescript", "react", "node.js",
    "docker", "kubernetes", "aws", "azure", "gcp", "machine learning", "nlp",
    "deep learning", "spacy", "fastapi", "flask", "django", "sql", "postgresql",
    "mongodb", "git", "linux", "html", "css", "pytorch", "tensorflow",
    "scikit-learn", "pandas", "numpy", "powerbi", "tableau", "spark", "hadoop",
    "agile", "scrum", "ci/cd", "rest api", "graphql", "go", "rust", "ruby", "php",
    
    # Business, Marketing, Management
    "marketing", "fundraising", "sales", "seo", "sem", "content creation", 
    "social media", "public relations", "b2b", "b2c", "lead generation", 
    "project management", "product management", "leadership", "communication",
    "budgeting", "strategy", "copywriting"
}

def extract_skills(text: str) -> list[str]:
    """
    Extract skills from text by matching tokens against a predefined taxonomy (SKILL_DB).
    Normalizes inputs to lowercase.
    """
    text_lower = text.lower()
    found_skills = set()
    
    # Semantic Ontology Overrides
    ONTOLOGY_EXPANSION = {
        "python libraries": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "matplotlib", "seaborn"],
        "machine learning": ["scikit-learn", "xgboost", "random forest"],
        "deep learning": ["tensorflow", "pytorch", "keras", "neural networks"],
        "mern": ["mongodb", "express.js", "react", "node.js"],
        "nlp": ["transformers", "spacy", "natural language processing"],
        "web tools": ["git", "github", "vs code"],
    }
    
    for skill in SKILL_DB:
        escaped_skill = re.escape(skill)
        
        # Word boundary handling for special characters 
        if skill.endswith("+") or skill.endswith("#"):
            pattern = r"\b" + escaped_skill + r"(?!\w)"
        else:
            pattern = r"\b" + escaped_skill + r"\b"
            
        if re.search(pattern, text_lower):
            found_skills.add(skill)
            
    # Sub-string match for broad categories bridging into specific missing technicals
    for broad_term, specific_skills in ONTOLOGY_EXPANSION.items():
        if broad_term in text_lower:
            found_skills.update(specific_skills)

    return list(found_skills)

def extract_skills_from_jd(jd_text: str) -> list[str]:
    """
    Wrapper for extracting required skills directly from the Job Description.
    """
    return extract_skills(jd_text)


def extract_required_skills(requirements_text: str) -> tuple[list[str], list[str]]:
    """
    Parses bullet lists or comma-separated text from a specific requirements block.
    Returns:
        raw_required_skills: List of raw skill strings extracted.
        normalized_required_skills: List of skills matched against SKILL_DB.
    """
    raw_skills = []
    
    # Check for bullets vs comma separated
    if "\n" in requirements_text or "-" in requirements_text or "*" in requirements_text:
        # Split by newlines, then remove bullets
        lines = requirements_text.split("\n")
        for line in lines:
            # strip common bullets and spaces
            cleaned = re.sub(r'^[\s\*\-\•\>]+\s*', '', line).strip()
            if cleaned:
                # further split by comma if mixed
                if "," in cleaned:
                    parts = [p.strip() for p in cleaned.split(",")]
                    raw_skills.extend([p for p in parts if p])
                else:
                    raw_skills.append(cleaned)
    else:
        # Comma-separated
        raw_skills = [s.strip() for s in requirements_text.split(",") if s.strip()]
        
    normalized = normalize_required_skills(raw_skills)
    
    return raw_skills, normalized

def normalize_required_skills(raw_skills: list[str]) -> list[str]:
    """
    Normalizes raw extracted skill strings by matching them against SKILL_DB.
    """
    normalized_skills = set()
    for raw in raw_skills:
        raw_lower = raw.lower()
        for skill in SKILL_DB:
            escaped_skill = re.escape(skill)
            if skill.endswith("+") or skill.endswith("#"):
                pattern = r"\b" + escaped_skill + r"(?!\w)"
            else:
                pattern = r"\b" + escaped_skill + r"\b"
            
            if re.search(pattern, raw_lower):
                normalized_skills.add(skill)
                
    return list(normalized_skills)
