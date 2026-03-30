import sys
import os

# Ensure the parent app path is available to absolute imports cleanly
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, BASE_DIR)

from app.services.scorer import scorer_service
import json

def evaluate_pipeline():
    """
    Manually crafted tests to validate exactly what the user requested:
    "Data Scientist", "NLP Engineer", "Full Stack Developer", "DevOps Engineer"
    """
    profiles = [
        {
            "id": 1,
            "target_role": "Data Scientist",
            "reqs": "Python, Machine Learning, scikit-learn",
            "jd": "We are seeking a senior machine learning expert with deep python knowledge.",
            "resume_text": "Experienced programmer strong in Python and Machine Learning. Built an NLP chatbot project using Transformers."
        },
        {
            "id": 2,
            "target_role": "NLP Engineer",
            "reqs": "NLP, PyTorch, Transformers",
            "jd": "NLP Engineer required to build conversational agents using advanced transformers.",
            "resume_text": "Experienced programmer strong in Python and Machine Learning. Built an NLP chatbot project using Transformers."
        },
        {
            "id": 3,
            "target_role": "Full Stack Developer",
            "reqs": "Javascript, React, Node.js, MERN",
            "jd": "Looking for a MERN stack expert to build consumer-facing REST APIs.",
            "resume_text": "Experienced frontend and backend developer. Created a full MERN project utilizing REST APIs."
        },
        {
            "id": 4,
            "target_role": "DevOps Engineer",
            "reqs": "AWS, Docker, Kubernetes, CI/CD",
            "jd": "Maintain our infrastructure using Kubernetes, Docker, and CI/CD pipelines in AWS.",
            "resume_text": "Experienced programmer strong in Python and Machine Learning. Built an NLP chatbot project using Transformers."
        }
    ]
    
    print("\n" + "="*50)
    print("🧠 EVALUATION SUITE: HYBRID SCORING ENGINE")
    print("="*50)

    for p in profiles:
        result = scorer_service.score_candidate(
            resume_text=p["resume_text"],
            job_role=p["target_role"],
            requirements=p["reqs"],
            job_description=p["jd"]
        )
        
        print(f"\n[Test #{p['id']}] vs '{p['target_role']}'")
        print(f"Resume Sample Snippet: {p['resume_text'][:60]}...")
        print(f"---- SCORING BREAKDOWN ----")
        print(f"|  Role Predicted:   {result['predicted_resume_role']}")
        print(f"|  Role Alignment:   {result['role_alignment_score']}")
        print(f"|  Semantic Score:   {result['semantic_similarity_score']}")
        print(f"|  Skill Match:      {result['skill_match_score']}")
        print(f"|  - Matched Skills: {', '.join(result['matched_skills']) if result['matched_skills'] else 'None'}")
        print(f"|  - Missing Skills: {', '.join(result['missing_skills']) if result['missing_skills'] else 'None'}")
        print(f"|  Final AI Score:   {result['final_score']}")
        print(f"|  Recommendation:   {result['recommendation']}")
        print("-" * 27)

if __name__ == "__main__":
    evaluate_pipeline()
