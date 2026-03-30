import re

def clean_text(text: str) -> str:
    """
    Cleans unstructured resume or job post text safely without destroying meaning.
    Preserves technical tokens like C++, Node.js, React.js.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase for uniform processing
    text = text.lower()
    
    # 2. Re-map non-standard whitespace and breaks
    text = re.sub(r'\\n|\\r|\\t', ' ', text)
    
    # 3. Handle structural patterns
    text = re.sub(r'[\*\•\-\■\●]', ' ', text)  # Common bullet points
    
    # 4. Remove generic punctuation that does NOT form technical terms.
    # We want to keep '+' (C++), '#' (C#), '.' (Node.js) if they are connected to words.
    # Replace anything that isn't a word char, space, +, #, or . inside a word.
    # Note: re.sub(r'[^\w\s\+#\.]', ' ', text) leaves trailing dots, so we do it carefully:
    text = re.sub(r'[^\w\s\+#\.\/]', ' ', text)
    
    # 5. Clean standalone trailing/leading puncts left over
    text = re.sub(r'(?<!\w)[\.\+](?!\w)', ' ', text)
    
    # 6. Normalize structural links / emails out to just domain names or remove
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # 7. Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_skill(skill: str) -> str:
    """ Standardize common permutations of tech skills. """
    skill_cleaned = skill.lower().strip()
    
    aliases = {
        "js": "javascript",
        "react.js": "react",
        "reactjs": "react",
        "node.js": "nodejs",
        "node": "nodejs",
        "vue.js": "vue",
        "scikit-learn": "sklearn",
        "scikit": "sklearn",
        "tf": "tensorflow",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "nlp": "natural language processing",
        "c++": "cpp",
        "k8s": "kubernetes",
        "gcp": "google cloud",
        "aws": "amazon web services",
        "ds": "data science"
    }
    
    return aliases.get(skill_cleaned, skill_cleaned)
