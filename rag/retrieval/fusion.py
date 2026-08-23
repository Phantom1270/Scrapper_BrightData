"""
Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.
"""

from typing import List, Tuple
from collections import defaultdict
from rag.models.retrieval import RetrievalResult


class ReciprocalRankFusion:
    def __init__(self, settings=None, k: int = None):
        if isinstance(settings, int):
            k = settings
            settings = None
            
        if k is None:
            if settings is None:
                from rag.config.settings import get_settings
                settings = get_settings()
            k = settings.retrieval.rrf_k
            
        self.k = k

    def fuse(self, ranked_lists: List[List[Tuple[str, float]]],
             weights: List[float] = None) -> List[Tuple[str, float]]:
        """
        Combine multiple ranked lists into one fused ranking using RRF.
        
        Args:
            ranked_lists: A list of ranked lists. Each inner list contains
                (item_id, score) tuples ordered by relevance (best first).
            weights: Optional list of weights for each ranked list.
                Must be the same length as ranked_lists.
                
        Returns:
            List[Tuple[str, float]]: Sorted list of fused items by score descending.
        """
        if not ranked_lists:
            return []
            
        if all(not lst for lst in ranked_lists):
            return []
            
        if weights is not None and len(weights) != len(ranked_lists):
            raise ValueError("Length of weights must match length of ranked_lists")
            
        if weights is None:
            weights = [1.0] * len(ranked_lists)
            
        fused_scores = defaultdict(float)
        
        for lst_idx, lst in enumerate(ranked_lists):
            weight = weights[lst_idx]
            for rank, (item_id, _) in enumerate(lst):
                # RRF formula: weight * 1 / (k + rank + 1)
                fused_scores[item_id] += weight * (1.0 / (self.k + rank + 1))
                
        # Sort by score descending
        fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def fuse_vector_and_bm25(self, vector_results: List[RetrievalResult],
                             bm25_results: List[RetrievalResult],
                             vector_weight: float = 0.6,
                             bm25_weight: float = 0.4) -> List[Tuple[str, float]]:
        """
        Convenience method for fusing vector and BM25 RetrievalResult lists.
        """
        vec_list = [(r.chunk_id, r.score) for r in vector_results]
        bm25_list = [(r.chunk_id, r.score) for r in bm25_results]
        
        return self.fuse([vec_list, bm25_list], weights=[vector_weight, bm25_weight])
