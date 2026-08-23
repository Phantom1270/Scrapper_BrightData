"""
Embeddings package for converting text to vector representations.
"""

from rag.search.embeddings.base import BaseEmbedder
from rag.search.embeddings.sentence_transformer import SentenceTransformerEmbedder

__all__ = [
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
]
