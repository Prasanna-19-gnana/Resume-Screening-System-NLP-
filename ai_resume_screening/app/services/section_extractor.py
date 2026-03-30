import re
import logging

logger = logging.getLogger(__name__)

def extract_sections(pdf_text: str) -> dict:
    """
    Robust regex+heuristic section extraction that survives PDF spacing formatting flaws.
    """
    sections = {
        "summary": "",
        "skills": "",
        "projects": "",
        "experience": "",
        "education": "",
        "certifications": "",
        "full_text": pdf_text
    }
    
    # We use re.IGNORECASE, allow leading spaces, bullets, numbers
    # We do NOT use \b because 'Skills:' might be parsed with random symbols around it
    boundaries = {
        "skills": r"^\s*[\W_]*\s*(technical\s+)?skills\s*|^[\W_]*\s*core competencies",
        "projects": r"^\s*[\W_]*\s*(personal\s+|academic\s+)?projects\s*",
        "experience": r"^\s*[\W_]*\s*(work\s+|professional\s+)?experience|employment history",
        "education": r"^\s*[\W_]*\s*education|academic background|qualifications",
        "certifications": r"^\s*[\W_]*\s*certifications|licenses"
    }
    
    current_sec = "summary"
    lines = pdf_text.split("\n")
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        # Headers are usually short
        if len(line_clean) < 50:
            matched_section = None
            for sec, pattern in boundaries.items():
                if re.match(pattern, line_clean, flags=re.IGNORECASE):
                    matched_section = sec
                    break
            
            if matched_section:
                current_sec = matched_section
                logger.info(f"Detected boundary switch -> {current_sec}: {line_clean}")
                continue # don't append the header string itself
                
        # Append line
        sections[current_sec] += line_clean + "\n"
        
    for k in sections:
        sections[k] = sections[k].strip()
        
    return sections
