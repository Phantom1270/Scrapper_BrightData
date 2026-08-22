"""
Document deduplicator.

Removes exact duplicates and near-duplicates from a list of
NormalizedDocuments before storage and indexing.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List

from rag.models.document import NormalizedDocument

logger = logging.getLogger(__name__)


class DocumentDeduplicator:
    """
    Remove duplicate and near-duplicate NormalizedDocuments.

    Two passes:
    1. Exact dedup  — MD5 hash-based, O(n)
    2. Near-dedup   — MinHash + LSH, O(n)  (falls back to exact-only
                      if datasketch is not installed)

    Args:
        similarity_threshold: Jaccard similarity threshold for near-dedup.
            Documents with similarity above this are considered duplicates.
            Default 0.85 works well for single-domain scraped data.
    """

    def __init__(self, settings=None, similarity_threshold: float = None) -> None:
        if isinstance(settings, float):
            similarity_threshold = settings
            settings = None

        if similarity_threshold is None:
            similarity_threshold = 0.85

        self._threshold = similarity_threshold
        self._stats: dict = {}

        # Check if datasketch is available
        try:
            from datasketch import MinHash, MinHashLSH  # noqa: F401
            self._has_minhash = True
        except ImportError:
            logger.warning(
                "datasketch not installed — near-dedup will be skipped. "
                "Install with: pip install datasketch"
            )
            self._has_minhash = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deduplicate(self, documents: List[NormalizedDocument]) -> List[NormalizedDocument]:
        """
        Return a deduplicated list of documents.

        Stats are available via get_stats() after this call.
        """
        input_count = len(documents)

        # Pass 1: exact dedup (O(n))
        after_exact, exact_removed = self._exact_dedup(documents)

        # Pass 2: near-dedup with MinHash LSH (O(n)), or skip if unavailable
        if self._has_minhash and len(after_exact) > 1:
            after_near, near_removed = self._minhash_dedup(after_exact)
        else:
            after_near = after_exact
            near_removed = 0

        self._stats = {
            "input_count":   input_count,
            "exact_removed": exact_removed,
            "near_removed":  near_removed,
            "output_count":  len(after_near),
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
    # Pass 1: Exact dedup  — O(n)
    # ------------------------------------------------------------------

    def _exact_dedup(
        self,
        documents: List[NormalizedDocument],
    ) -> tuple[List[NormalizedDocument], int]:
        seen: Dict[str, NormalizedDocument] = {}

        for doc in documents:
            text = self._full_text(doc)
            h = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()

            if h not in seen:
                seen[h] = doc
            else:
                existing = seen[h]
                if len(doc.content_blocks) > len(existing.content_blocks):
                    seen[h] = doc

        removed = len(documents) - len(seen)
        return list(seen.values()), removed

    # ------------------------------------------------------------------
    # Pass 2: Near-dedup — O(n) with MinHash LSH
    # ------------------------------------------------------------------

    def _minhash_dedup(
        self,
        documents: List[NormalizedDocument],
    ) -> tuple[List[NormalizedDocument], int]:
        from datasketch import MinHash, MinHashLSH

        NUM_PERM = 128
        SHINGLE_SIZE = 3  # 3-word shingles

        lsh = MinHashLSH(threshold=self._threshold, num_perm=NUM_PERM)
        minhashes: List[MinHash] = []

        # Build all MinHashes first
        for i, doc in enumerate(documents):
            m = MinHash(num_perm=NUM_PERM)
            tokens = self._full_text(doc).split()
            if len(tokens) >= SHINGLE_SIZE:
                for j in range(len(tokens) - SHINGLE_SIZE + 1):
                    shingle = " ".join(tokens[j:j + SHINGLE_SIZE])
                    m.update(shingle.encode("utf-8"))
            else:
                # Very short doc — hash whole text
                m.update(self._full_text(doc).encode("utf-8"))
            minhashes.append(m)

        # Insert one-by-one; query before inserting to avoid self-matches
        duplicate_indices: set[int] = set()
        for i in range(len(documents)):
            if i in duplicate_indices:
                # Already marked as a duplicate — still insert so later docs
                # can be compared against it (keeps the first occurrence)
                lsh.insert(str(i), minhashes[i])
                continue

            candidates = lsh.query(minhashes[i])
            # Any already-inserted candidate that matches means this doc
            # is a near-duplicate of an earlier (kept) document
            if candidates:
                duplicate_indices.add(i)
            else:
                lsh.insert(str(i), minhashes[i])

        unique = [doc for i, doc in enumerate(documents) if i not in duplicate_indices]
        removed = len(duplicate_indices)
        return unique, removed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _full_text(doc: NormalizedDocument) -> str:
        """Concatenate all content block texts into one string."""
        parts = [doc.title, doc.description] if doc.title else [doc.description]
        parts += [b.text for b in doc.content_blocks]
        return "\n\n".join(p for p in parts if p)

