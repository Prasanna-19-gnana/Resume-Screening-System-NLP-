import json
import os
import uuid
from typing import List, Dict, Any

class DatasetBuilder:
    """
    Utility class to create, manage, and label a dataset for training the 
    resume scoring model.
    """
    def __init__(self, data_dir: str = "../../data"):
        self.data_dir = data_dir
        self.dataset_path = os.path.join(self.data_dir, "training_dataset.json")
        os.makedirs(self.data_dir, exist_ok=True)
        self.dataset = self.load_dataset()

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads the dataset from the JSON file if it exists."""
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_dataset(self):
        """Saves the local dataset list to the JSON file."""
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=4)
        print(f"Dataset saved to {self.dataset_path} with {len(self.dataset)} records.")

    def add_sample(
        self,
        job_description: str,
        resume_text: str,
        required_skills: List[str],
        preferred_skills: List[str],
        matched_skills_true: List[str] = None,
        missing_skills_true: List[str] = None,
        fit_label: str = None,
        fit_score_true: int = None
    ):
        """Adds a single fully/partially labeled sample to the dataset."""
        sample = {
            "id": str(uuid.uuid4()),
            "job_description": job_description,
            "resume_text": resume_text,
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "matched_skills_true": matched_skills_true or [],
            "missing_skills_true": missing_skills_true or [],
            "fit_label": fit_label or "unknown",  # good, medium, poor, unknown
            "fit_score_true": fit_score_true if fit_score_true is not None else 0  # 0 to 100
        }
        self.dataset.append(sample)
        self.save_dataset()

    def label_unlabeled_samples_cli(self):
        """A simple CLI helper to manually label samples that are lacking labels."""
        unlabeled = [s for s in self.dataset if s["fit_label"] == "unknown"]
        if not unlabeled:
            print("No unlabeled samples found!")
            return

        print(f"Found {len(unlabeled)} unlabeled samples. Starting manual annotation...")
        for sample in unlabeled:
            print("-" * 50)
            print("JOB REQUIRED SKILLS:", ", ".join(sample["required_skills"]))
            print("RESUME TEXT (Preview):", sample["resume_text"][:200], "...")
            
            matched = input("Enter true matched skills (comma separated): ")
            sample["matched_skills_true"] = [s.strip() for s in matched.split(",") if s.strip()]
            
            missing = input("Enter true missing skills (comma separated): ")
            sample["missing_skills_true"] = [s.strip() for s in missing.split(",") if s.strip()]
            
            fit = input("Enter fit label (good/medium/poor): ").strip().lower()
            if fit in ["good", "medium", "poor"]:
                sample["fit_label"] = fit
            
            score_str = input("Enter true fit score (0-100): ").strip()
            if score_str.isdigit():
                sample["fit_score_true"] = int(score_str)
                
        self.save_dataset()
        print("Annotation complete.")

# ---------------------------------------------------------
# Usage example: Generate initial sample data automatically
# ---------------------------------------------------------
if __name__ == "__main__":
    builder = DatasetBuilder(data_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data")))
    
    # Only add examples if the dataset is empty to prevent duplication on re-runs
    if len(builder.dataset) == 0:
        print("Generating mock data samples...")
        
        # Sample 1: Perfect match (Good)
        builder.add_sample(
            job_description="We need a strong Data Scientist with Python, PyTorch, and NLP experience. Experience with BERT is a must.",
            resume_text="Senior Machine Learning Engineer with 5 years of Python experience. I have built diverse NLP models using PyTorch and specifically tuned BERT for classification tasks.",
            required_skills=["Python", "PyTorch", "NLP", "BERT"],
            preferred_skills=["HuggingFace", "AWS"],
            matched_skills_true=["Python", "PyTorch", "NLP", "BERT"],
            missing_skills_true=[],
            fit_label="good",
            fit_score_true=95
        )

        # Sample 2: Partial match (Medium)
        builder.add_sample(
            job_description="Looking for a Backend Developer. Required: FastAPI, PostgreSQL, Docker.",
            resume_text="Backend Developer skilled in Flask, Python, and MySQL. Some exposure to Docker during college projects.",
            required_skills=["FastAPI", "PostgreSQL", "Docker"],
            preferred_skills=["Kubernetes", "Redis"],
            matched_skills_true=["Docker"],
            missing_skills_true=["FastAPI", "PostgreSQL"],
            fit_label="medium",
            fit_score_true=45
        )

        # Sample 3: Terrible match (Poor)
        builder.add_sample(
            job_description="Frontend Engineer skilled in React, TypeScript, and TailwindCSS.",
            resume_text="Civil Engineer with 10 years of experience managing construction projects. Excellent AutoCAD skills and team leadership.",
            required_skills=["React", "TypeScript", "TailwindCSS"],
            preferred_skills=["Next.js", "Figma"],
            matched_skills_true=[],
            missing_skills_true=["React", "TypeScript", "TailwindCSS"],
            fit_label="poor",
            fit_score_true=5
        )
