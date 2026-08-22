"""
Query request/response data models.

Phase 4.8 (Serving) uses these models as the query interface contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QueryRequest:
    """Incoming query from a user or API caller."""

    question: str
    top_k: int = 5
    filter_content_type: Optional[str] = None
    filter_doc_id: Optional[str] = None
    use_reranking: bool = True

    def __post_init__(self) -> None:
        if not self.question or not self.question.strip():
            raise ValueError("QueryRequest.question must not be empty.")
        if self.top_k < 1:
            raise ValueError("QueryRequest.top_k must be >= 1.")

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "top_k": self.top_k,
            "filter_content_type": self.filter_content_type,
            "filter_doc_id": self.filter_doc_id,
            "use_reranking": self.use_reranking,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QueryRequest":
        return cls(
            question=data["question"],
            top_k=data.get("top_k", 5),
            filter_content_type=data.get("filter_content_type"),
            filter_doc_id=data.get("filter_doc_id"),
            use_reranking=data.get("use_reranking", True),
        )


@dataclass
class QueryResponse:
    """Response returned after generation."""

    answer: str
    sources: List[dict]        # [{chunk_id, heading, url, score, content_type}]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "total_time_ms": self.total_time_ms,
            "cached": self.cached,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QueryResponse":
        return cls(
            answer=data["answer"],
            sources=data.get("sources") or [],
            retrieval_time_ms=data.get("retrieval_time_ms", 0.0),
            generation_time_ms=data.get("generation_time_ms", 0.0),
            total_time_ms=data.get("total_time_ms", 0.0),
            cached=data.get("cached", False),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueryResponse):
            return NotImplemented
        return self.to_dict() == other.to_dict()
