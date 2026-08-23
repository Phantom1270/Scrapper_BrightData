"""
Abstract storage interface (BaseStore).

All storage backends (SQLite, FileStore, future vector stores)
must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from rag.models.document import NormalizedDocument
from rag.models.chunk import Chunk


class BaseStore(ABC):
    """Abstract base class for RAG storage backends."""

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    @abstractmethod
    def save_documents(self, documents: List[NormalizedDocument]) -> None:
        """Persist a list of documents. Should upsert on doc_id collision."""
        ...

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[NormalizedDocument]:
        """Return a single document by doc_id, or None if not found."""
        ...

    @abstractmethod
    def get_all_documents(self) -> List[NormalizedDocument]:
        """Return every document in the store."""
        ...

    @abstractmethod
    def get_documents_by_template(self, template_id: str) -> List[NormalizedDocument]:
        """Return all documents produced by a given template (e.g. 'tpl_002')."""
        ...

    @abstractmethod
    def count_documents(self) -> int:
        """Return the total number of documents in the store."""
        ...

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    @abstractmethod
    def save_chunks(self, chunks: List[Chunk]) -> None:
        """Persist a list of chunks. Should upsert on chunk_id collision."""
        ...

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Return a single chunk by chunk_id, or None if not found."""
        ...

    @abstractmethod
    def get_chunks_by_doc(self, doc_id: str) -> List[Chunk]:
        """Return all chunks belonging to a document, ordered by chunk_index."""
        ...

    @abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        """Return every chunk in the store."""
        ...

    @abstractmethod
    def count_chunks(self) -> int:
        """Return the total number of chunks in the store."""
        ...

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    @abstractmethod
    def delete_all(self) -> None:
        """Delete all documents and chunks. Use with care."""
        ...
