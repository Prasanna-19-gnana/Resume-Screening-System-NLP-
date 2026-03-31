"""
RANKING EVALUATION: Compute metrics for resume ranking

Metrics:
- Precision@K: How many of top K are relevant
- Recall@K: How many relevant items are in top K
- NDCG@K: Normalized discounted cumulative gain
- MRR: Mean reciprocal rank
"""

import logging
from typing import List, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class RankingEvaluator:
    """Evaluate ranking quality"""
    
    @staticmethod
    def precision_at_k(
        predicted_ranks: List[int],
        relevant_indices: List[int],
        k: int = 5
    ) -> float:
        """
        Precision@K: proportion of top-k predictions that are relevant
        
        Args:
            predicted_ranks: Ranking order (indices)
            relevant_indices: Ground truth relevant indices
            k: Consider only top k results
        
        Returns:
            Precision@k (0-1)
        """
        
        relevant_set = set(relevant_indices)
        top_k = predicted_ranks[:k]
        
        if len(top_k) == 0:
            return 0.0
        
        n_relevant = sum(1 for idx in top_k if idx in relevant_set)
        precision = n_relevant / len(top_k)
        
        return precision
    
    @staticmethod
    def recall_at_k(
        predicted_ranks: List[int],
        relevant_indices: List[int],
        k: int = 5
    ) -> float:
        """
        Recall@K: proportion of all relevant items found in top-k
        
        Args:
            predicted_ranks: Ranking order (indices)
            relevant_indices: Ground truth relevant indices
            k: Consider only top k results
        
        Returns:
            Recall@k (0-1)
        """
        
        if len(relevant_indices) == 0:
            return 0.0
        
        relevant_set = set(relevant_indices)
        top_k = predicted_ranks[:k]
        
        n_relevant = sum(1 for idx in top_k if idx in relevant_set)
        recall = n_relevant / len(relevant_indices)
        
        return recall
    
    @staticmethod
    def ndcg_at_k(
        scores: List[float],
        relevance: List[int],
        k: int = 5
    ) -> float:
        """
        NDCG@K: Normalized discounted cumulative gain
        
        Measures ranking quality considering:
        1. Position discounting (higher positions weighted more)
        2. Relevance gradients (soft relevance)
        
        Args:
            scores: Predicted scores (used for ranking)
            relevance: Ground truth relevance labels (0-5)
            k: Consider only top k results
        
        Returns:
            NDCG@k (0-1)
        """
        
        assert len(scores) == len(relevance), "Scores and relevance must have same length"
        
        # Remove invalid scores
        valid_indices = [i for i, s in enumerate(scores) if not np.isnan(s)]
        if not valid_indices:
            return 0.0
        
        scores_valid = [scores[i] for i in valid_indices]
        relevance_valid = [relevance[i] for i in valid_indices]
        
        # Get ranking (sorted indices)
        ranking = sorted(
            range(len(scores_valid)),
            key=lambda i: scores_valid[i],
            reverse=True
        )[:k]
        
        # Compute DCG
        dcg = 0.0
        for position, idx in enumerate(ranking, 1):
            rel = relevance_valid[idx]
            dcg += rel / np.log2(position + 1)
        
        # Compute ideal DCG (perfect ranking)
        ideal_ranking = sorted(
            range(len(relevance_valid)),
            key=lambda i: relevance_valid[i],
            reverse=True
        )[:k]
        
        idcg = 0.0
        for position, idx in enumerate(ideal_ranking, 1):
            rel = relevance_valid[idx]
            idcg += rel / np.log2(position + 1)
        
        # Compute NDCG
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg
    
    @staticmethod
    def mean_reciprocal_rank(
        predicted_ranks: List[int],
        relevant_indices: List[int]
    ) -> float:
        """
        MRR: Mean reciprocal rank
        
        Measures how high the first relevant item is ranked
        
        Args:
            predicted_ranks: Ranking order (indices)
            relevant_indices: Ground truth relevant indices
        
        Returns:
            MRR (0-1)
        """
        
        relevant_set = set(relevant_indices)
        
        for position, idx in enumerate(predicted_ranks, 1):
            if idx in relevant_set:
                return 1.0 / position
        
        return 0.0
    
    @staticmethod
    def average_precision(
        predicted_ranks: List[int],
        relevant_indices: List[int]
    ) -> float:
        """
        Average Precision: Average of precision at each relevant position
        
        Args:
            predicted_ranks: Ranking order (indices)
            relevant_indices: Ground truth relevant indices
        
        Returns:
            AP (0-1)
        """
        
        relevant_set = set(relevant_indices)
        
        if len(relevant_set) == 0:
            return 0.0
        
        precisions = []
        n_relevant_so_far = 0
        
        for position, idx in enumerate(predicted_ranks, 1):
            if idx in relevant_set:
                n_relevant_so_far += 1
                precision_at_position = n_relevant_so_far / position
                precisions.append(precision_at_position)
        
        if len(precisions) == 0:
            return 0.0
        
        average_precision = sum(precisions) / len(relevant_set)
        return average_precision
    
    @staticmethod
    def evaluate_ranking(
        predicted_scores: List[float],
        ground_truth_scores: List[float],
        k_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive ranking evaluation
        
        Args:
            predicted_scores: Model predictions
            ground_truth_scores: Ground truth scores
            k_values: List of K values to evaluate (default: [1, 5, 10])
        
        Returns:
            Evaluation metrics dict
        """
        
        if k_values is None:
            k_values = [1, 5, 10]
        
        # Convert to relevance (1 if above threshold, 0 otherwise)
        threshold = np.median(ground_truth_scores)
        relevance = [1 if score >= threshold else 0 for score in ground_truth_scores]
        
        # Get predicted ranking
        predicted_ranking = sorted(
            range(len(predicted_scores)),
            key=lambda i: predicted_scores[i],
            reverse=True
        )
        
        # Get ground truth relevant indices
        relevant_indices = [i for i, r in enumerate(relevance) if r == 1]
        
        results = {
            "n_items": len(predicted_scores),
            "n_relevant": len(relevant_indices),
            "threshold": threshold
        }
        
        # Compute metrics for each K
        for k in k_values:
            k_str = str(k)
            results[f"precision@{k_str}"] = RankingEvaluator.precision_at_k(
                predicted_ranking, relevant_indices, k
            )
            results[f"recall@{k_str}"] = RankingEvaluator.recall_at_k(
                predicted_ranking, relevant_indices, k
            )
            results[f"ndcg@{k_str}"] = RankingEvaluator.ndcg_at_k(
                predicted_scores, relevance, k
            )
        
        # Overall metrics
        results["mrr"] = RankingEvaluator.mean_reciprocal_rank(
            predicted_ranking, relevant_indices
        )
        results["ap"] = RankingEvaluator.average_precision(
            predicted_ranking, relevant_indices
        )
        
        return results
    
    @staticmethod
    def print_metrics(metrics: Dict[str, Any]) -> None:
        """Pretty print evaluation metrics"""
        
        print("\n" + "="*50)
        print("RANKING EVALUATION METRICS")
        print("="*50)
        
        print(f"\nDataset: {metrics['n_items']} candidates, {metrics['n_relevant']} relevant")
        print(f"Threshold: {metrics['threshold']:.2f}")
        
        print("\nPrecision:")
        for k in [1, 5, 10]:
            k_str = str(k)
            if f"precision@{k_str}" in metrics:
                p = metrics[f"precision@{k_str}"]
                print(f"  P@{k_str}: {p:.4f}")
        
        print("\nRecall:")
        for k in [1, 5, 10]:
            k_str = str(k)
            if f"recall@{k_str}" in metrics:
                r = metrics[f"recall@{k_str}"]
                print(f"  R@{k_str}: {r:.4f}")
        
        print("\nNDCG:")
        for k in [1, 5, 10]:
            k_str = str(k)
            if f"ndcg@{k_str}" in metrics:
                ndcg = metrics[f"ndcg@{k_str}"]
                print(f"  NDCG@{k_str}: {ndcg:.4f}")
        
        if "mrr" in metrics:
            print(f"\nMRR: {metrics['mrr']:.4f}")
        if "ap" in metrics:
            print(f"AP: {metrics['ap']:.4f}")
        
        print("="*50 + "\n")
