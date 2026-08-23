"""
Chunk data model.

Phase 4.3 (Chunking Engine) produces Chunk instances.
Chunks are the atomic unit for indexing and retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """
    A single retrieval-ready chunk derived from a NormalizedDocument.

    heading_path forms a breadcrumb trail, e.g.:
        ["API Reference", "config_context", "Parameters"]

    content_with_heading prepends this breadcrumb to the content text
    so that the embedding captures structural context.
    """

    chunk_id: str
    doc_id: str
    url: str
    content: str
    content_type: str
    heading_path: List[str]
    chunk_index: int
    token_count: int
    parent_chunk_id: Optional[str] = None
    block_type: str = ""
    language: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("Chunk.chunk_id must not be empty.")
        if not self.doc_id:
            raise ValueError("Chunk.doc_id must not be empty.")
        if self.chunk_index < 0:
            raise ValueError("Chunk.chunk_index must be >= 0.")
        if self.token_count < 0:
            raise ValueError("Chunk.token_count must be >= 0.")
        if self.metadata is None:
            self.metadata = {}
        if self.heading_path is None:
            self.heading_path = []

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def content_with_heading(self) -> str:
        """
        Return the chunk content prefixed with its heading breadcrumb.

        Example:
            "## API Reference > config_context > Parameters\n\nThe actual chunk text..."
        """
        if self.heading_path:
            heading_str = " > ".join(self.heading_path)
            return f"## {heading_str}\n\n{self.content}"
        return self.content

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "url": self.url,
            "content": self.content,
            "content_type": self.content_type,
            "heading_path": self.heading_path,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "parent_chunk_id": self.parent_chunk_id,
            "block_type": self.block_type,
            "language": self.language,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Deserialize from a dict (as produced by to_dict)."""
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            url=data["url"],
            content=data["content"],
            content_type=data.get("content_type", "prose"),
            heading_path=data.get("heading_path") or [],
            chunk_index=data["chunk_index"],
            token_count=data.get("token_count", 0),
            parent_chunk_id=data.get("parent_chunk_id"),
            block_type=data.get("block_type", ""),
            language=data.get("language", ""),
            metadata=data.get("metadata") or {},
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.to_dict() == other.to_dict()
