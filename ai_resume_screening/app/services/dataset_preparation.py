"""
DATASET PREPARATION: Create training data for ML model

Strategy:
1. Load training data (resume-JD pairs)
2. Extract features using FeatureEngineer
3. Generate labels using current scorer (pseudo-labels)
4. Package as (X, y) for model training
"""

import json
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepare training dataset for ML model"""
    
    def __init__(self, feature_engineer, current_scorer=None):
        """
        Initialize with feature engineer and optional current scorer
        
        Args:
            feature_engineer: FeatureEngineer instance
            current_scorer: Current scorer for generating pseudo-labels (optional)
        """
        self.feature_engineer = feature_engineer
        self.current_scorer = current_scorer
    
    def prepare_dataset_from_samples(
        self,
        samples: List[Dict[str, str]]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Prepare dataset from sample data
        
        Args:
            samples: List of dicts with keys:
                - "resume_text": str
                - "job_description": str
                - "requirements": str
                - "score": float (optional, for ground truth)
        
        Returns:
            X: Feature matrix (n_samples, 7)
            y: Label vector (n_samples,)
            metadata: List of dicts with sample info
        """
        
        X_list = []
        y_list = []
        metadata = []
        
        for idx, sample in enumerate(samples):
            try:
                resume_text = sample.get("resume_text", "")
                job_description = sample.get("job_description", "")
                requirements = sample.get("requirements", "")
                
                # Extract features
                features = self.feature_engineer.extract_features(
                    resume_text=resume_text,
                    job_description=job_description,
                    requirements=requirements
                )
                
                # Generate label: use provided score or current scorer
                if "score" in sample:
                    label = sample["score"] / 100.0  # Normalize to 0-1
                else:
                    # Use current scorer to generate pseudo-label
                    # Simplified approach: compute a score from features
                    # This avoids complex scorer interface
                    try:
                        # Pseudo-label from features (simple heuristic)
                        ext = features.get("skills_exact_match_ratio", 0.5)
                        par = features.get("skills_partial_match_ratio", 0.3)
                        sem = features.get("semantic_similarity_score", 0.5)
                        role = features.get("role_alignment_score", 0.5)
                        
                        # Weighted combination: 45% skills, 35% semantic, 20% role
                        pseudo_score = (ext + par) * 0.45 + sem * 0.35 + role * 0.20
                        label = min(1.0, max(0.0, pseudo_score))
                    except:
                        label = 0.5
                
                # Add to lists
                feature_vector = [
                    features["semantic_similarity_score"],
                    features["skills_exact_match_ratio"],
                    features["skills_partial_match_ratio"],
                    features["role_alignment_score"],
                    features["number_of_matched_skills"],
                    features["number_of_missing_skills"],
                    features["top_sentence_similarity"]
                ]
                X_list.append(feature_vector)
                y_list.append(label)
                
                # Store metadata
                meta = {
                    "idx": idx,
                    "score": label * 100,  # Convert back to 0-100
                    "features": features,
                    "resume_preview": resume_text[:100] if resume_text else ""
                }
                metadata.append(meta)
                
                logger.info(f"Sample {idx}: score={label*100:.1f}")
            
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Dataset prepared: {len(X)} samples, {X.shape[1]} features")
        
        return X, y, metadata
    
    def prepare_dataset_from_file(
        self,
        file_path: str
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Load dataset from JSON or CSV file
        
        Args:
            file_path: Path to training data file
        
        Returns:
            X, y, metadata
        """
        
        file_path = Path(file_path)
        
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    samples = data.get("samples", [data])
                else:
                    samples = data
        
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            samples = df.to_dict('records')
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return self.prepare_dataset_from_samples(samples)
    
    def create_synthetic_dataset(self, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic training data from test cases
        
        Uses predefined test samples to generate training data
        """
        
        # Test samples with various skill levels
        test_samples = [
            {
                "role": "NLP Engineer",
                "score": 92,
                "resume_snippet": "Python NLP expertise with transformers, BERT, spaCy, seq2seq models, semantic search"
            },
            {
                "role": "Full Stack Developer",
                "score": 45,
                "resume_snippet": "React, Node.js, MongoDB, REST APIs, some machine learning exposure"
            },
            {
                "role": "Data Analyst",
                "score": 30,
                "resume_snippet": "Excel, SQL, Tableau, some Python, no NLP experience"
            },
            {
                "role": "Junior Software Engineer",
                "score": 25,
                "resume_snippet": "C++, Java, basic Python, no machine learning or NLP"
            },
            {
                "role": "ML Engineer",
                "score": 85,
                "resume_snippet": "PyTorch, TensorFlow, scikit-learn, deep learning, some NLP projects"
            }
        ]
        
        # Job description
        job_description = """
        We are looking for an experienced NLP Engineer to join our AI team.
        
        Requirements:
        - 5+ years of Python development
        - Strong expertise in NLP: transformers, BERT, GPT
        - Experience with semantic search and similarity
        - Deep learning frameworks: PyTorch or TensorFlow
        - Spacy and NLTK experience
        - Machine learning best practices
        - Experience with large language models
        - Vector embeddings and similarity search
        """
        
        requirements = "Python, NLP, transformers, BERT, semantic search, PyTorch, spaCy, embedding"
        
        # Generate variations by replicating and perturbing
        samples = []
        for sample in test_samples:
            for i in range(n_samples // len(test_samples)):
                samples.append({
                    "resume_text": sample["resume_snippet"],
                    "job_description": job_description,
                    "requirements": requirements,
                    "score": sample["score"]
                })
        
        X, y, metadata = self.prepare_dataset_from_samples(samples)
        
        logger.info(f"Synthetic dataset: {len(X)} samples")
        return X, y
    
    def split_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Split dataset into train/val/test
        
        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        
        n = len(X)
        indices = np.random.permutation(n)
        
        test_idx = int(n * test_size)
        val_idx = int(n * (test_size + validation_size))
        
        test_indices = indices[:test_idx]
        val_indices = indices[test_idx:val_idx]
        train_indices = indices[val_idx:]
        
        return (
            (X[train_indices], y[train_indices]),
            (X[val_indices], y[val_indices]),
            (X[test_indices], y[test_indices])
        )
    
    def save_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str
    ) -> None:
        """Save features and labels to disk"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(str(output_path / "X_train.npy"), X)
        np.save(str(output_path / "y_train.npy"), y)
        
        logger.info(f"Dataset saved to {output_path}")
    
    def load_dataset(self, input_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load features and labels from disk"""
        
        input_path = Path(input_dir)
        
        X = np.load(str(input_path / "X_train.npy"))
        y = np.load(str(input_path / "y_train.npy"))
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
