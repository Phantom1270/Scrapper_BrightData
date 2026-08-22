"""
Document deduplicator.

Removes exact duplicates and near-duplicates from a list of
NormalizedDocuments before storage and indexing.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from rag.models.document import NormalizedDocument
from rag.utils.hashing import content_hash, near_duplicate_ratio

logger = logging.getLogger(__name__)


class DocumentDeduplicator:
    """
    Remove duplicate and near-duplicate NormalizedDocuments.

    Two passes:
    1. Exact dedup — hash-based, O(n)
    2. Near-dedup  — pairwise SequenceMatcher within the same content_type, O(n²)

    The near-dedup pass is bounded to the same content_type to avoid
    false positives (an api_reference doc and a tutorial can legitimately
    share large blocks of text but be different pages).
    """

    def __init__(self, settings=None, similarity_threshold: float = None) -> None:
        if isinstance(settings, float):
            similarity_threshold = settings
            settings = None
            
        if similarity_threshold is None:
            similarity_threshold = 0.92
            
        self._threshold = similarity_threshold
        self._stats: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deduplicate(self, documents: List[NormalizedDocument]) -> List[NormalizedDocument]:
        """
        Return a deduplicated list of documents.

        Processes in two passes (exact then near-duplicate).
        Stats are available via get_stats() after this call.
        """
        input_count = len(documents)
        after_exact, exact_removed = self._exact_dedup(documents)
        after_near, near_removed = self._near_dedup(after_exact)

        self._stats = {
            "input_count":    input_count,
            "exact_removed":  exact_removed,
            "near_removed":   near_removed,
            "output_count":   len(after_near),
        }
        logger.info(
            "Dedup: %d → %d (exact=%d, near=%d)",
            input_count, len(after_near), exact_removed, near_removed,
        )
        return after_near

    def get_stats(self) -> dict:
        """Return stats from the last deduplicate() call."""
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Pass 1: Exact dedup
    # ------------------------------------------------------------------

    def _exact_dedup(
        self,
        documents: List[NormalizedDocument],
    ) -> tuple[List[NormalizedDocument], int]:
        """
        Hash-based exact deduplication.

        For collisions, keep the document with more content blocks
        (treat that as the more-complete version).
        """
        seen: Dict[str, NormalizedDocument] = {}   # hash → doc

        for doc in documents:
            full_text = self._full_text(doc)
            h = content_hash(full_text)

            if h not in seen:
                seen[h] = doc
            else:
                existing = seen[h]
                # Keep whichever has more content blocks
                if len(doc.content_blocks) > len(existing.content_blocks):
                    seen[h] = doc

        removed = len(documents) - len(seen)
        return list(seen.values()), removed

    # ------------------------------------------------------------------
    # Pass 2: Near-duplicate dedup
    # ------------------------------------------------------------------

    def _near_dedup(
        self,
        documents: List[NormalizedDocument],
    ) -> tuple[List[NormalizedDocument], int]:
        """
        Near-duplicate detection using SequenceMatcher, scoped per content_type.

        For each document, compare against already-accepted documents of the
        SAME content_type. If similarity > threshold, discard the new one.
        """
        # Group accepted docs by content_type for efficient comparison
        accepted_by_type: Dict[str, List[tuple[str, NormalizedDocument]]] = {}
        # tuple = (full_text, doc)
        all_accepted: List[NormalizedDocument] = []
        removed = 0

        for doc in documents:
            ct = doc.content_type
            full_text = self._full_text(doc)

            is_near_dupe = False
            for accepted_text, _ in accepted_by_type.get(ct, []):
                ratio = near_duplicate_ratio(full_text, accepted_text)
                if ratio > self._threshold:
                    is_near_dupe = True
                    break

            if is_near_dupe:
                removed += 1
                continue

            # Accept this doc
            all_accepted.append(doc)
            accepted_by_type.setdefault(ct, []).append((full_text, doc))

        return all_accepted, removed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _full_text(doc: NormalizedDocument) -> str:
        """Concatenate all content block texts into one string."""
        parts = [doc.title, doc.description] if doc.title else [doc.description]
        parts += [b.text for b in doc.content_blocks]
        return "\n\n".join(p for p in parts if p)
