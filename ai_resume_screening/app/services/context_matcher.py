"""
CONTEXT-AWARE MATCHING: Sentence-Level Semantic Similarity
- Splits resume into sentences
- Computes similarity scores for each sentence vs job description
- Extracts evidence (top matching sentences)
- Identifies weak/missing areas
"""

import logging
from typing import List, Dict, Tuple, Any
import re
from sentence_transformers import SentenceTransformer, util
import torch

logger = logging.getLogger(__name__)


class ContextAwareMatcher:
    """
    Performs sentence-level semantic matching between resume and job description
    """
    
    def __init__(self):
        """Initialize with sentence transformer model"""
        try:
            # Use lightweight model for fast inference
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.model = None
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, filtering empty ones
        """
        if not text:
            return []
        
        # Split on common sentence delimiters
        sentences = re.split(r'[.!?]\s+', text)
        
        # Clean and filter
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            # Keep sentences with at least 5 words
            if len(sent.split()) >= 5:
                cleaned.append(sent)
        
        return cleaned
    
    def compute_sentence_similarities(
        self,
        resume_text: str,
        job_description: str
    ) -> Dict[str, Any]:
        """
        Compute similarity of each resume sentence to job description
        
        Returns:
        {
            "sentences": [list of resume sentences],
            "similarities": [list of similarity scores],
            "top_matches": [
                {"sentence": str, "score": float, "rank": int},
                ...
            ],
            "avg_similarity": float,
            "max_similarity": float,
            "coverage_score": float  # % of sentences with >0.3 similarity
        }
        """
        
        if not self.model:
            logger.error("Model not loaded, cannot compute similarities")
            return {
                "sentences": [],
                "similarities": [],
                "top_matches": [],
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "coverage_score": 0.0
            }
        
        # Split resume into sentences
        resume_sentences = self.split_into_sentences(resume_text)
        
        if not resume_sentences:
            return {
                "sentences": [],
                "similarities": [],
                "top_matches": [],
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "coverage_score": 0.0
            }
        
        # Encode all sentences
        resume_embeddings = self.model.encode(resume_sentences, convert_to_tensor=True)
        jd_embedding = self.model.encode(job_description, convert_to_tensor=True)
        
        # Compute cosine similarities
        similarities = util.pytorch_cos_sim(resume_embeddings, jd_embedding)
        similarities = similarities.squeeze().cpu().numpy()
        
        # Handle single sentence case (returns scalar instead of array)
        if isinstance(similarities, float):
            similarities = [similarities]
        else:
            similarities = similarities.tolist()
        
        # Find top matches
        top_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True
        )[:3]  # Top 3
        
        top_matches = []
        for rank, idx in enumerate(top_indices, 1):
            if similarities[idx] > 0.1:  # Only include if reasonable score
                top_matches.append({
                    "sentence": resume_sentences[idx],
                    "score": float(similarities[idx]),
                    "rank": rank
                })
        
        # Compute metrics
        avg_similarity = float(sum(similarities) / len(similarities))
        max_similarity = float(max(similarities))
        
        # Coverage: % of sentences with meaningful similarity (>0.3)
        meaningful = sum(1 for s in similarities if s > 0.3)
        coverage_score = meaningful / len(resume_sentences) if resume_sentences else 0.0
        
        return {
            "sentences": resume_sentences,
            "similarities": similarities,
            "top_matches": top_matches,
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "coverage_score": coverage_score
        }
    
    def extract_evidence(
        self,
        resume_text: str,
        job_description: str,
        num_evidence: int = 3
    ) -> Dict[str, Any]:
        """
        Extract strong matching evidence from resume
        
        Returns:
        {
            "strong_matches": [list of top sentences with scores],
            "weak_areas": [list of weak sentences],
            "missing_coverage": float,  # % of resume not matching
            "evidence_summary": str  # Brief narrative summary
        }
        """
        
        sim_result = self.compute_sentence_similarities(resume_text, job_description)
        
        # Extract weak areas (low similarity sentences)
        weak_areas = []
        for i, sent in enumerate(sim_result["sentences"]):
            if sim_result["similarities"][i] < 0.2:
                weak_areas.append({
                    "sentence": sent,
                    "score": float(sim_result["similarities"][i])
                })
        
        # Priority: weak areas that contain important job keywords
        job_keywords = extract_keywords_from_jd(job_description)
        weak_with_keywords = []
        for weak in weak_areas:
            keywords_found = sum(
                1 for kw in job_keywords
                if kw.lower() in weak["sentence"].lower()
            )
            if keywords_found > 0:
                weak_with_keywords.append(weak)
        
        # Take top weak areas if any
        weak_with_keywords = sorted(
            weak_with_keywords,
            key=lambda x: x["score"],
            reverse=True
        )[:3]  # Top 3 weak areas
        
        # Missing coverage
        missing_coverage = 1.0 - sim_result["coverage_score"]
        
        # Evidence summary
        if sim_result["top_matches"]:
            evidence_count = len(sim_result["top_matches"])
            evidence_summary = f"{evidence_count} strong matching areas found. "
            if weak_with_keywords:
                evidence_summary += f"{len(weak_with_keywords)} areas need improvement."
            else:
                evidence_summary += "Good coverage overall."
        else:
            evidence_summary = "Limited matching evidence found. Resume may not align with job requirements."
        
        return {
            "strong_matches": sim_result["top_matches"],
            "weak_areas": weak_with_keywords,
            "missing_coverage": missing_coverage,
            "evidence_summary": evidence_summary,
            "avg_similarity": sim_result["avg_similarity"],
            "coverage_score": sim_result["coverage_score"]
        }


def extract_keywords_from_jd(job_description: str) -> List[str]:
    """
    Extract important keywords from job description
    """
    # Common job posting keywords to exclude
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'are', 'be', 'been', 'have',
        'has', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
        'must', 'may', 'might', 'can', 'your', 'you', 'we', 'our', 'their',
        'this', 'that', 'these', 'those', 'experience', 'knowledge', 'skills'
    }
    
    # Split into words and clean
    words = re.findall(r'\b\w+\b', job_description.lower())
    
    # Filter
    keywords = []
    for word in words:
        if len(word) > 3 and word not in stopwords:
            keywords.append(word)
    
    # Return unique keywords
    return list(set(keywords))
