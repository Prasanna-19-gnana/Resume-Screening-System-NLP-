import json
import os
import numpy as np
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("Warning: sentence-transformers not installed. Semantic similarity will be 0.")

class FeatureExtractor:
    """
    Converts raw dataset samples (JSON format) into numeric feature vectors 
    ready for machine learning models.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            # We load a small, fast sentence transformer for semantic features
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            self.model = None
            print(f"Could not load SentenceTransformer: {e}")

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        if not self.model or not text1 or not text2:
            return 0.0
        
        # Compute embeddings
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        
        # Compute cosine similarity
        sim = util.pytorch_cos_sim(emb1, emb2).item()
        return max(0.0, sim)  # ensure non-negative

    def extract_features(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Extracts numeric features from a single resume-JD pair.
        """
        # 1. Skill Coverage Features
        req_skills = set(s.lower() for s in sample.get("required_skills", []))
        pref_skills = set(s.lower() for s in sample.get("preferred_skills", []))
        
        # Assuming matched_skills_true contains all correctly identified skills for this sample
        matched_skills = set(s.lower() for s in sample.get("matched_skills_true", []))
        
        # Calculate overlap
        matched_req = req_skills.intersection(matched_skills)
        matched_pref = pref_skills.intersection(matched_skills)
        
        req_coverage = len(matched_req) / len(req_skills) if req_skills else 0.0
        pref_coverage = len(matched_pref) / len(pref_skills) if pref_skills else 0.0
        
        # Missing required count
        # Could also use sample["missing_skills_true"] if rigorously annotated
        missing_req_count = len(req_skills) - len(matched_req)
        
        total_matched = len(matched_skills)

        # 2. Semantic Similarity
        # In a more advanced version, you might split the JD into 'Responsibilities' 
        # and compare against the Resume's 'Experience' section.
        semantic_sim = self.compute_semantic_similarity(
            sample.get("job_description", ""), 
            sample.get("resume_text", "")
        )
        
        # 3. Resume Quality Signals
        resume_text = sample.get("resume_text", "")
        resume_length = len(resume_text.split())
        
        # Construct the feature vector
        features = {
            "req_skill_coverage": float(req_coverage),
            "pref_skill_coverage": float(pref_coverage),
            "total_matched_skills": float(total_matched),
            "missing_req_skills": float(missing_req_count),
            "semantic_similarity": float(semantic_sim),
            "resume_word_count": float(resume_length)
        }
        
        return features

    def process_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Reads the dataset JSON and appends a 'features' dict to each row."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        processed_data = []
        for sample in data:
            print(f"Extracting features for sample: {sample.get('id')}...")
            row = sample.copy()
            row["features"] = self.extract_features(sample)
            processed_data.append(row)
            
        return processed_data

# ---------------------------------------------------------
# Usage example: Test extraction on created dataset
# ---------------------------------------------------------
if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    dataset_path = os.path.join(data_dir, "training_dataset.json")
    
    extractor = FeatureExtractor()
    
    try:
        enriched_dataset = extractor.process_dataset(dataset_path)
        
        # Save the dataset with features included to a new file for training
        output_path = os.path.join(data_dir, "training_dataset_with_features.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_dataset, f, indent=4)
            
        print(f"\nSuccessfully extracted features for {len(enriched_dataset)} samples.")
        print(f"Saved enriched data to {output_path}")
        
        # Print an example feature vector
        print("\nExample Feature Vector for Sample 1:")
        print(json.dumps(enriched_dataset[0]["features"], indent=4))
        
    except FileNotFoundError as e:
        print(e)
