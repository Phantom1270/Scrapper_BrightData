"""
Abstract base class for all embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Abstract interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of document/chunk texts.
        Returns a list of embedding vectors, one per input text.
        Order must match input order.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single search query.
        May apply a different prefix/strategy than embed_documents.
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Return the embedding dimension. 
        Must be consistent across all embeddings from this provider.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return a string identifier for the model being used.
        Stored in the index manifest for reproducibility.
        """
        pass

    def embed_documents_batched(self, texts: List[str], batch_size: int = None, show_progress: bool = True) -> List[List[float]]:
        """
        Convenience method that calls embed_documents in batches.
        """
        if not texts:
            return []

        if batch_size is None:
            # Try to get it from settings, or default to 64
            try:
                from rag.config.settings import get_settings
                settings = get_settings()
                batch_size = settings.embedding.batch_size
            except Exception:
                batch_size = 64

        all_embeddings = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            if show_progress:
                print(f"Embedded {len(all_embeddings)}/{total} chunks...")
                
        return all_embeddings
