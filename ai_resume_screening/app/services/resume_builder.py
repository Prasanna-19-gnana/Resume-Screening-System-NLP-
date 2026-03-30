import re
import logging

logger = logging.getLogger(__name__)

def build_smart_resume_text(parsed_sections: dict) -> str:
    """
    Builds a high quality semantic text input for sentence embeddings!
    Restricts context to skills + projects + experience. (No education / certifications)
    Removes semantic duplicate phrases.
    """
    
    # Aggregate only relevant technical sections
    blocks = [
        parsed_sections.get("skills", ""),
        parsed_sections.get("projects", ""),
        parsed_sections.get("experience", "")
    ]
    
    raw_combined = "\n".join([b for b in blocks if b]).strip()
    
    # Simple semantic deduplication trick for repeated keywords (SentenceTransformers prefer dense logic)
    # E.g. If "machine learning" appears 10 times, we don't need all 10 to establish vector location.
    # However, keeping paragraphs intact is better. We map out explicitly identical lines.
    
    lines = raw_combined.split("\n")
    unique_lines = []
    seen = set()
    
    for line in lines:
        clean_l = line.strip().lower()
        if len(clean_l) > 5 and clean_l not in seen:
            unique_lines.append(line.strip())
            seen.add(clean_l)
            
    # Reassemble
    smart_text = "\n".join(unique_lines)
    
    # Limit tokens strictly to avoid exceeding transformer bounds (~512 to 1500 words chunked)
    words = smart_text.split()
    if len(words) > 800:
        logger.info(f"Clipping overly long technical history: {len(words)} -> 800")
        smart_text = " ".join(words[:800])
        
    logger.info(f"Constructed Smart Resume Vector Text Length: {len(smart_text.split())} words")
    
    return smart_text
