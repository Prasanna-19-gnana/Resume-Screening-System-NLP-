import os
import pandas as pd
import random

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_DIR = os.path.join(BASE_DIR, "data/")
os.makedirs(DATA_DIR, exist_ok=True)

GOLD_DATASET_PATH = os.path.join(DATA_DIR, "custom_gold_dataset.csv")

def generate_custom_gold_dataset(num_samples=150):
    print("🚀 Generating Custom Gold Dataset...")
    
    # Template Profiles
    data = []
    
    profiles = [
        {"resume_profile": "ML student", "skills": "python, machine learning, scikit-learn, pytorch", "job_domain": "Data"},
        {"resume_profile": "MERN dev", "skills": "javascript, react, node.js, express, mongodb", "job_domain": "WebDev"},
        {"resume_profile": "NLP student", "skills": "python, nlp, transformers, huggingface, pytorch", "job_domain": "Data"},
        {"resume_profile": "DevOps Engineer", "skills": "aws, docker, kubernetes, terraform, linux", "job_domain": "Infrastructure"},
        {"resume_profile": "Data Analyst", "skills": "sql, excel, powerbi, tableau, python", "job_domain": "Data"},
        {"resume_profile": "Recent Grad", "skills": "java, c++, python, algorithms", "job_domain": "General"}
    ]
    
    jobs = [
        {"role": "Data Scientist", "jd": "Looking for a Data Scientist to build predictive models. Requires Python, Machine Learning, and SQL.", "domain": "Data"},
        {"role": "NLP Engineer", "jd": "We need an NLP Engineer to work with LLMs and Transformers. Required: Python, PyTorch, HuggingFace.", "domain": "Data"},
        {"role": "Fullstack Web Developer", "jd": "Seeking a developer skilled in React and Node.js for our web applications.", "domain": "WebDev"},
        {"role": "Cloud Architect", "jd": "Looking for DevOps/Cloud experience with AWS, Kubernetes, and CI/CD.", "domain": "Infrastructure"},
        {"role": "Business Analyst", "jd": "Identify business trends using SQL, Excel, and Tableau.", "domain": "Data"}
    ]
    
    for _ in range(num_samples):
        candidate = random.choice(profiles)
        target_job = random.choice(jobs)
        
        # Heuristic scoring logic for gold dataset
        score = 0
        reason = ""
        
        # Base domain match
        if candidate["job_domain"] == target_job["domain"]:
            # Same domain, check specific relevance
            if candidate["resume_profile"] == "ML student" and target_job["role"] == "Data Scientist":
                score = 75
                reason = "has ML but lacks specific production tools"
            elif candidate["resume_profile"] == "NLP student" and target_job["role"] == "NLP Engineer":
                score = 85
                reason = "strong match with domain and tools"
            elif candidate["resume_profile"] == "MERN dev" and target_job["role"] == "Fullstack Web Developer":
                score = 90
                reason = "perfect stack alignment"
            elif candidate["resume_profile"] == "DevOps Engineer" and target_job["role"] == "Cloud Architect":
                score = 80
                reason = "infrastructure tooling matches well"
            elif candidate["resume_profile"] == "Data Analyst" and target_job["role"] == "Business Analyst":
                score = 85
                reason = "analytics tools strongly match"
            elif candidate["resume_profile"] == "Data Analyst" and target_job["role"] == "Data Scientist":
                score = 40
                reason = "lacks core machine learning modeling skills"
            else:
                score = 60
                reason = "domain match but specialized mismatch"
        else:
            # Different domain
            if candidate["resume_profile"] == "MERN dev" and target_job["role"] == "Data Scientist":
                score = 30
                reason = "wrong domain (web dev applying for ML)"
            elif candidate["resume_profile"] == "Recent Grad":
                score = 45
                reason = "general skills, lacks specific domain depth"
            else:
                score = random.randint(20, 35)
                reason = f"completely irrelevant background ({candidate['job_domain']} to {target_job['domain']})"
                
        data.append({
            "Resume_Profile": candidate["resume_profile"],
            "Resume_Skills": candidate["skills"],
            "Job_Role": target_job["role"],
            "Job_Description": target_job["jd"],
            "Target_Score": score,
            "Reason": reason
        })

    df = pd.DataFrame(data)
    df.to_csv(GOLD_DATASET_PATH, index=False)
    print(f"✅ Generated {len(df)} rows of Golden Data.")
    print(f"💾 Saved to: {GOLD_DATASET_PATH}")
    
    print("\nSAMPLE:")
    print(df.head(5).to_markdown())

if __name__ == "__main__":
    generate_custom_gold_dataset(150)
