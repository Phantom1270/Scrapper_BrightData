"""
Data models for normalized documents.

Phase 4.2 (Data Pipeline) produces NormalizedDocument instances.
These are the canonical representation of a scraped page before chunking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# ContentBlock
# ---------------------------------------------------------------------------


@dataclass
class ContentBlock:
    """
    A single block of content within a document.

    block_type values:
        "prose"              — Plain paragraph text
        "code"               — Code snippet
        "table"              — Tabular data
        "parameter_list"     — List of parameters (name/type/description)
        "function_signature" — Function/method signature
        "note"               — Note, warning, tip callout
        "example"            — Usage example (may contain code)
        "unknown"            — Could not determine type
    """

    block_type: str
    text: str
    heading: str = ""
    language: str = ""
    structured_data: Optional[dict] = None

    def __post_init__(self) -> None:
        if not self.block_type:
            raise ValueError("ContentBlock.block_type must not be empty.")
        valid_types = {
            "prose", "code", "table", "parameter_list",
            "function_signature", "note", "example", "unknown",
        }
        if self.block_type not in valid_types:
            raise ValueError(
                f"ContentBlock.block_type={self.block_type!r} is not valid. "
                f"Choose from: {sorted(valid_types)}"
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "block_type": self.block_type,
            "text": self.text,
            "heading": self.heading,
            "language": self.language,
            "structured_data": self.structured_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContentBlock":
        """Deserialize from a dict (as produced by to_dict)."""
        return cls(
            block_type=data["block_type"],
            text=data["text"],
            heading=data.get("heading", ""),
            language=data.get("language", ""),
            structured_data=data.get("structured_data"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContentBlock):
            return NotImplemented
        return self.to_dict() == other.to_dict()


# ---------------------------------------------------------------------------
# NormalizedDocument
# ---------------------------------------------------------------------------


@dataclass
class NormalizedDocument:
    """
    Canonical representation of a single scraped page.

    Produced by Phase 4.2, consumed by Phase 4.3 (chunking).

    content_type values:
        "api_reference"  — API doc page
        "tutorial"       — Step-by-step guide
        "notebook"       — Jupyter-style example
        "example"        — Short code example
        "unknown"        — Could not determine
    """

    doc_id: str
    url: str
    title: str
    description: str
    content_blocks: List[ContentBlock]
    metadata: dict
    template_id: str
    content_type: str
    source_link: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("NormalizedDocument.doc_id must not be empty.")
        if not self.url:
            raise ValueError("NormalizedDocument.url must not be empty.")

        valid_content_types = {
            "api_reference", "tutorial", "notebook", "example", "unknown",
        }
        if self.content_type not in valid_content_types:
            raise ValueError(
                f"NormalizedDocument.content_type={self.content_type!r} is not valid. "
                f"Choose from: {sorted(valid_content_types)}"
            )

        # Ensure mutable default is never shared
        if self.metadata is None:
            self.metadata = {}
        if self.content_blocks is None:
            self.content_blocks = []

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def full_text(self) -> str:
        """Return all block texts joined with double newlines."""
        return "\n\n".join(b.text for b in self.content_blocks if b.text)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "content_blocks": [b.to_dict() for b in self.content_blocks],
            "metadata": self.metadata,
            "template_id": self.template_id,
            "content_type": self.content_type,
            "source_link": self.source_link,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NormalizedDocument":
        """Deserialize from a dict (as produced by to_dict)."""
        raw_blocks = data.get("content_blocks") or []
        blocks = [ContentBlock.from_dict(b) for b in raw_blocks]
        return cls(
            doc_id=data["doc_id"],
            url=data["url"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            content_blocks=blocks,
            metadata=data.get("metadata") or {},
            template_id=data.get("template_id", ""),
            content_type=data.get("content_type", "unknown"),
            source_link=data.get("source_link"),
            error=data.get("error"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NormalizedDocument):
            return NotImplemented
        return self.to_dict() == other.to_dict()
