"""
RetrievalResult data model.

Phase 4.5 (Retrieval Engine) produces RetrievalResult instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RetrievalResult:
    """
    A single result returned by the retrieval engine.

    source values:
        "vector"   — came from vector similarity search
        "bm25"     — came from BM25 keyword search
        "hybrid"   — combined score from both
        "reranked" — passed through a cross-encoder re-ranker
    """

    chunk_id: str
    content: str
    url: str
    heading_path: List[str]
    content_type: str
    score: float
    source: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_sources = {"vector", "bm25", "hybrid", "reranked"}
        if self.source not in valid_sources:
            raise ValueError(
                f"RetrievalResult.source={self.source!r} is not valid. "
                f"Choose from: {sorted(valid_sources)}"
            )
        if self.metadata is None:
            self.metadata = {}
        if self.heading_path is None:
            self.heading_path = []

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "url": self.url,
            "heading_path": self.heading_path,
            "content_type": self.content_type,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalResult":
        return cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            url=data["url"],
            heading_path=data.get("heading_path") or [],
            content_type=data.get("content_type", "prose"),
            score=data["score"],
            source=data["source"],
            metadata=data.get("metadata") or {},
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RetrievalResult):
            return NotImplemented
        return self.to_dict() == other.to_dict()
