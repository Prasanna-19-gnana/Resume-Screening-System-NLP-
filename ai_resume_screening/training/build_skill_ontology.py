import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
ONTOLOGY_PATH = os.path.join(BASE_DIR, "data", "processed", "skill_ontology.json")

def build_ontology():
    """
    Design as a maintainable Python dictionary.
    Includes partial credit logic map: e.g. 'nlp' expands to 'transformers', 'chatbot'.
    """
    ontology = {
        "machine learning": [
            "scikit-learn", "sklearn", "xgboost", "pandas", "numpy", 
            "tensorflow", "pytorch", "keras", "ml"
        ],
        "deep learning": [
            "tensorflow", "pytorch", "keras", "neural networks", "cnn", "rnn"
        ],
        "nlp": [
            "transformers", "tokenization", "text classification", 
            "chatbot", "spacy", "huggingface", "bert", "gpt",
            "natural language processing"
        ],
        "web development": [
            "rest api", "frontend", "backend", "html", "css", 
            "javascript", "typescript", "django", "flask", "fastapi"
        ],
        "mern": [
            "mongodb", "express", "react", "nodejs", "node.js", "react.js"
        ],
        "devops": [
            "docker", "kubernetes", "aws", "gcp", "azure", 
            "ci/cd", "jenkins", "terraform", "bash", "linux"
        ],
        "data science": [
            "sql", "python", "r", "machine learning", "tableau", 
            "powerbi", "statistics", "data analysis"
        ]
    }
    
    os.makedirs(os.path.dirname(ONTOLOGY_PATH), exist_ok=True)
    with open(ONTOLOGY_PATH, "w") as f:
        json.dump(ontology, f, indent=4)
        
    print(f"✅ Skill Ontology successfully built and serialized to => {ONTOLOGY_PATH}")
    print(f"Total parent taxonomy categories: {len(ontology.keys())}")

if __name__ == "__main__":
    build_ontology()
