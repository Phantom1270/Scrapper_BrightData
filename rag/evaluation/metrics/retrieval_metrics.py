"""
Metrics for measuring retrieval quality.
"""

from typing import List
import math


class RetrievalMetrics:
    """Calculates retrieval metrics like Precision, Recall, MRR, nDCG."""

    def precision_at_k(self, retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
        """Of the top-k retrieved chunks, what fraction are relevant?"""
        if k == 0 or not retrieved_ids:
            return 0.0
            
        top_k = retrieved_ids[:k]
        relevant_count = sum(1 for chunk_id in top_k if chunk_id in expected_ids)
        
        return relevant_count / k

    def recall_at_k(self, retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
        """Of all expected chunks, what fraction were in the top-k?"""
        if not expected_ids:
            return 0.0
            
        if k == 0 or not retrieved_ids:
            return 0.0
            
        top_k = retrieved_ids[:k]
        relevant_count = sum(1 for chunk_id in top_k if chunk_id in expected_ids)
        
        return relevant_count / len(expected_ids)

    def mean_reciprocal_rank(self, retrieved_ids: List[str], expected_ids: List[str]) -> float:
        """1 / rank of the first relevant result."""
        if not expected_ids or not retrieved_ids:
            return 0.0
            
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in expected_ids:
                return 1.0 / (i + 1)
                
        return 0.0

    def ndcg_at_k(self, retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        if not expected_ids or not retrieved_ids or k == 0:
            return 0.0
            
        # Calculate DCG
        dcg = 0.0
        for i in range(min(k, len(retrieved_ids))):
            if retrieved_ids[i] in expected_ids:
                rel_i = 1.0
                dcg += rel_i / math.log2(i + 2)
                
        # Calculate Ideal DCG
        idcg = 0.0
        ideal_count = min(k, len(expected_ids))
        for i in range(ideal_count):
            idcg += 1.0 / math.log2(i + 2)
            
        if idcg == 0.0:
            return 0.0
            
        return dcg / idcg

    def evaluate_retrieval(self, retrieved_ids: List[str], expected_ids: List[str], k_values: List[int] = None) -> dict:
        """Compute all metrics at multiple k values."""
        if k_values is None:
            k_values = [1, 3, 5, 10]
            
        metrics = {
            "retrieved_count": len(retrieved_ids),
            "expected_count": len(expected_ids),
            "overlap_count": sum(1 for chunk_id in retrieved_ids if chunk_id in expected_ids),
            "mrr": self.mean_reciprocal_rank(retrieved_ids, expected_ids),
        }
        
        for k in k_values:
            metrics[f"precision@{k}"] = self.precision_at_k(retrieved_ids, expected_ids, k)
            metrics[f"recall@{k}"] = self.recall_at_k(retrieved_ids, expected_ids, k)
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(retrieved_ids, expected_ids, k)
            
        return metrics
