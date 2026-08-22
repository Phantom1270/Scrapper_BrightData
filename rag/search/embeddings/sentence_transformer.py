"""
SentenceTransformer embedding provider.
"""

from typing import List

from rag.search.embeddings.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using local sentence-transformers models."""

    def __init__(self, settings=None):
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()
            
        self.model_name = settings.embedding.model_name
        self.device = settings.embedding.device
        self.batch_size = settings.embedding.batch_size

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Handle bge model query prefix
        if "bge" in self.model_name.lower():
            self.query_prefix = "Represent this sentence for searching relevant passages: "
            self.doc_prefix = ""
        else:
            self.query_prefix = ""
            self.doc_prefix = ""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed document chunks."""
        if not texts:
            return []
            
        # Prepend doc_prefix if any
        if self.doc_prefix:
            processed_texts = [self.doc_prefix + text for text in texts]
        else:
            processed_texts = texts
            
        # sentence-transformers outputs numpy arrays, we convert to list of floats
        embeddings = self.model.encode(
            processed_texts, 
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        processed_query = self.query_prefix + query
        embedding = self.model.encode(
            processed_query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def get_model_name(self) -> str:
        """Return the model identifier."""
        return self.model_name
