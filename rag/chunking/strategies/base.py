"""
Abstract base class for all chunking strategies.
"""

from abc import ABC, abstractmethod
from typing import List
import tiktoken

from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument
from rag.utils.ids import generate_chunk_id
from rag.utils.tokens import count_tokens


class BaseChunkingStrategy(ABC):
    """Base strategy for chunking NormalizedDocuments."""

    def __init__(self, settings=None, config=None):
        if config is not None:
            settings = config
            
        if isinstance(settings, dict):
            config = settings
        else:
            if settings is None:
                from rag.config.settings import get_settings
                settings = get_settings()
            config = {
                "max_tokens": settings.chunking.max_tokens,
                "min_tokens": settings.chunking.min_tokens,
                "overlap_tokens": settings.chunking.overlap_tokens,
                "encoding_name": settings.chunking.encoding_name,
                "parent_max_tokens": getattr(settings.chunking, "parent_max_tokens", 1500)
            }
            
        self.config = config
        self.max_tokens = config.get("max_tokens", 512)
        self.min_tokens = config.get("min_tokens", 100)
        self.overlap_tokens = config.get("overlap_tokens", 75)
        self.encoding_name = config.get("encoding_name", "cl100k_base")
        self.encoding = tiktoken.get_encoding(self.encoding_name)

    @abstractmethod
    def chunk(self, document: NormalizedDocument) -> List[Chunk]:
        """Takes a normalized document, returns a list of Chunks."""
        pass

    def _split_text_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int, encoding_name: str) -> List[str]:
        """
        Token-bounded text splitting with overlap.
        Breaks at sentence boundaries if possible.
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = start_idx + max_tokens
            if end_idx >= len(tokens):
                chunks.append(self.encoding.decode(tokens[start_idx:]))
                break

            # Decode the current window to find a sentence boundary in the second half
            window_tokens = tokens[start_idx:end_idx]
            window_text = self.encoding.decode(window_tokens)
            
            # Second half of the window
            half_len = len(window_text) // 2
            
            # Try to find a boundary
            break_idx = -1
            for boundary in ["\n\n", ".\n", ". "]:
                idx = window_text.rfind(boundary, half_len)
                if idx != -1:
                    break_idx = idx + len(boundary)
                    break
            
            if break_idx != -1:
                # We found a boundary, re-encode just the part up to the boundary to know how many tokens to advance
                actual_text = window_text[:break_idx]
                actual_tokens = self.encoding.encode(actual_text)
                chunks.append(actual_text)
                
                # Advance start_idx by the number of tokens we actually consumed, minus overlap
                advance = len(actual_tokens) - overlap_tokens
                # Ensure we make progress
                if advance <= 0:
                    advance = 1
                start_idx += advance
            else:
                # No boundary found, split at max_tokens
                chunks.append(window_text)
                advance = max_tokens - overlap_tokens
                if advance <= 0:
                    advance = 1
                start_idx += advance

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Shortcut to token counting."""
        return count_tokens(text, self.encoding_name)

    def _make_chunk(
        self, 
        doc: NormalizedDocument, 
        content: str,
        content_type: str, 
        heading_path: List[str],
        chunk_index: int, 
        block_type: str = "",
        language: str = "", 
        parent_chunk_id: str = None,
        metadata: dict = None
    ) -> Chunk:
        """Helper to construct a Chunk object."""
        return Chunk(
            chunk_id=generate_chunk_id(doc.doc_id, chunk_index),
            doc_id=doc.doc_id,
            url=doc.url,
            content=content,
            content_type=content_type,
            heading_path=heading_path,
            chunk_index=chunk_index,
            token_count=self._count_tokens(content),
            parent_chunk_id=parent_chunk_id,
            block_type=block_type,
            language=language,
            metadata=metadata or {}
        )
