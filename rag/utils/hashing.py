"""
Content hashing utilities for deduplication.

Used to detect exact and near-duplicate content across documents
before indexing, so we don't embed the same text twice.
"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher


def content_hash(text: str) -> str:
    """
    Compute a normalized content hash for deduplication.

    Normalizes text before hashing:
    1. Lowercase
    2. Collapse all whitespace runs to a single space
    3. Strip leading/trailing whitespace

    Returns the first 16 characters of the SHA-256 hex digest.

    Args:
        text: Input text (any length).

    Returns:
        16-character hex string.
    """
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def near_duplicate_ratio(text_a: str, text_b: str) -> float:
    """
    Fast near-duplicate detection between two texts.

    Compares the first 500 characters and the last 500 characters of
    each text using SequenceMatcher. This is intentionally fast — it
    does not compare the full text for large documents.

    Args:
        text_a: First text string.
        text_b: Second text string.

    Returns:
        Similarity ratio between 0.0 (completely different)
        and 1.0 (identical). Texts are considered near-duplicates
        when the ratio is >= 0.92 (configurable by the caller).
    """
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    # Build comparison fingerprints from start + end windows
    window = 500
    sample_a = text_a[:window] + text_a[-window:]
    sample_b = text_b[:window] + text_b[-window:]

    return SequenceMatcher(None, sample_a, sample_b, autojunk=False).ratio()
