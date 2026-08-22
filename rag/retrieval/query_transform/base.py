"""
Abstract base class for query transformers.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseQueryTransformer(ABC):
    """Abstract interface for query transformers."""

    @abstractmethod
    def transform(self, query: str) -> List[str]:
        """Transform a query into one or more search queries.
        Always includes the original query in the output.
        Returns a list of query strings (at least 1 element)."""
        pass

    @abstractmethod
    def get_transform_name(self) -> str:
        """Return a string identifier for this transformer."""
        pass
