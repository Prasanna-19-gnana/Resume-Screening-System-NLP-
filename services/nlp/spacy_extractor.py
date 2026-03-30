import re
import spacy

# Load small english model as default. 
# In a real setup, we would ensure it's downloaded during deployment.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback message; usually handled before running the API
    print("Warning: en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")

def segment_sections(text: str) -> dict:
    """
    Heuristic-based section segmentation for resumes.
    Splits text into Skills, Experience, Education, etc. using Regex and line boundaries.
    """
    sections = {
        "summary": "",
        "skills": "",
        "experience": "",
        "education": "",
        "projects": ""
    }
    
    # Define common keywords that denote section headers in resumes
    boundaries = {
        "skills": r"\b(skills|technical skills|core competencies)\b",
        "experience": r"\b(experience|work experience|professional experience|employment history)\b",
        "education": r"\b(education|academic background|qualifications)\b",
        "projects": r"\b(projects|personal projects|academic projects)\b"
    }
    
    current_sec = "summary"
    lines = text.split("\n")
    
    for line in lines:
        line_clean = line.strip().lower()
        if not line_clean:
            continue
            
        # Check if line looks like a header (short length, matches keywords)
        if len(line_clean) < 40:
            matched_section = None
            for sec, pattern in boundaries.items():
                if re.match(pattern, line_clean):
                    matched_section = sec
                    break
            
            # If a new section is found, switch the active section pointer
            if matched_section:
                current_sec = matched_section
                continue
                
        # Append line to the currently active section
        sections[current_sec] += line + "\n"
        
    # Clean up excess whitespace
    return {k: v.strip() for k, v in sections.items()}
