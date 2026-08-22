"""
Abstract base class for all vector store providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict


class BaseVectorStore(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    def add_chunks(self, chunk_ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict]) -> None:
        """
        Add chunks to the vector store.
        If a chunk_id already exists, update it (upsert behavior).
        chunk_ids, embeddings, documents, and metadatas are parallel lists.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 10, content_type_filter: str = None, doc_id_filter: str = None) -> List[Dict]:
        """
        Search for similar chunks.
        Returns list of dicts: [{"id": str, "score": float, "document": str, "metadata": dict}, ...]
        Score semantics: higher = more similar (cosine similarity).
        Filters are optional — if provided, only return matching chunks.
        """
        pass

    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Remove specific chunks by ID."""
        pass

    @abstractmethod
    def delete_all(self) -> None:
        """Remove all chunks from the store."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of chunks in the store."""
        pass

    @abstractmethod
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a single chunk by ID.
        Returns dict with keys: id, document, metadata, embedding (if available).
        Returns None if not found.
        """
        pass
