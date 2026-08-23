"""
Deterministic ID generation for documents and chunks.

IDs are derived from content, so re-processing the same URL always
produces the same ID — enabling safe re-runs and deduplication.
"""

from __future__ import annotations

import hashlib


def generate_doc_id(url: str, index: int = 0) -> str:
    """
    Generate a deterministic 16-character hex ID for a document.

    The ID is derived from SHA-256 of "{url}::{index}", truncated to 16 chars.
    The index parameter allows multiple documents from the same URL
    (e.g. paginated API responses).

    Args:
        url:   The canonical URL of the document.
        index: Optional disambiguator (default 0).

    Returns:
        16-character hex string, e.g. "a3f2b1c4d5e6f789".
    """
    raw = f"{url}::{index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Generate a deterministic 16-character hex ID for a chunk.

    The ID is derived from SHA-256 of "{doc_id}::chunk::{chunk_index}".

    Args:
        doc_id:      Parent document ID.
        chunk_index: 0-based position of the chunk within the document.

    Returns:
        16-character hex string.
    """
    raw = f"{doc_id}::chunk::{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
