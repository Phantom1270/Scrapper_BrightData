"""
Abstract base class for re-rankers.
"""

from abc import ABC, abstractmethod
from typing import List
from rag.models.retrieval import RetrievalResult


class BaseReranker(ABC):
    """Abstract interface for re-rankers."""

    @abstractmethod
    def rerank(self, query: str, candidates: List[RetrievalResult],
               top_k: int = 5) -> List[RetrievalResult]:
        """Re-rank candidates by relevance to the query.
        Returns top_k results sorted by reranker score (best first).
        Updates each RetrievalResult's score and source fields."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return a string identifier for the reranker model."""
        pass
