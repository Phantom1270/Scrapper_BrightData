"""
Cross-encoder re-ranking using sentence-transformers.
"""

from typing import List
from rag.retrieval.reranker.base import BaseReranker
from rag.models.retrieval import RetrievalResult


class CrossEncoderReranker(BaseReranker):
    """Re-ranker using sentence-transformers CrossEncoder."""

    def __init__(self, settings=None):
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()
            
        model_name = settings.reranker.model_name
        device = settings.embedding.device

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )

        self.model = CrossEncoder(model_name, device=device)
        self.model_name_str = model_name

    def rerank(self, query: str, candidates: List[RetrievalResult],
               top_k: int = 5) -> List[RetrievalResult]:
        """Re-rank candidates using the cross-encoder model."""
        if not candidates:
            return []

        pairs = [(query, candidate.content) for candidate in candidates]
        
        # CrossEncoder predicts a score for each pair
        scores = self.model.predict(pairs, batch_size=32)
        
        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)
            candidate.source = "reranked"
            
        # Sort by the new score descending
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates[:top_k]

    def get_model_name(self) -> str:
        """Return a string identifier for the reranker model."""
        return self.model_name_str
