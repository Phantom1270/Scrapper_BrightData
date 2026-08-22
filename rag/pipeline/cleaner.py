"""
Text cleaning pipeline (Phase 4.2).

Wraps Phase 4.1's utils/text.py functions and adds pipeline-specific
cleaning strategies per ContentBlock type.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import replace
from typing import Optional

from rag.models.document import ContentBlock, NormalizedDocument
from rag.utils.text import (
    clean_unicode,
    clean_whitespace,
    strip_boilerplate,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regexes (compiled once at import time for performance)
# ---------------------------------------------------------------------------

_RE_COPY_CLIPBOARD = re.compile(r"\bcopy\s+to\s+clipboard\b", re.IGNORECASE)
_RE_PROGRESS_CHARS = re.compile(r"[━╸─│█▉▊▋▌▍▎▏▐░▒▓╔╗╚╝╠╣╦╩╬]{2,}")
_RE_SOURCE_MARKER = re.compile(r"\[source\]\s*#?", re.IGNORECASE)
_RE_TRAILING_HASH = re.compile(r"\s+#+\s*$")
_RE_HTML_TAG = re.compile(r"<[^>]{1,100}>")
_RE_PUNCT_ONLY = re.compile(r"^[\s\W]+$")
# Line-number prefix patterns: "1: ", " 1  ", ">>> "
_RE_LINE_NUMBER_PREFIX = re.compile(r"^(?:\d+:\s+|>>> |\.\.\. )", re.MULTILINE)
# Parameters section inline-dump marker
_RE_PARAM_DUMP = re.compile(r"\s*Parameters\s*:", re.MULTILINE)
_RE_YIELDS_DUMP = re.compile(r"\s*Yields\s*:", re.MULTILINE)

# -------- Title artifact patterns --------
_RE_TITLE_REPEATED = re.compile(r"(.{50,}?)\1{2,}")


class ContentCleaner:
    """
    Apply type-aware cleaning to ContentBlocks and NormalizedDocuments.

    Different block types require fundamentally different strategies:
    - Code: preserve whitespace, remove artifacts only
    - Prose/Description/Note: full text normalization
    - Signature: collapse spaces, strip markers, optionally truncate
    - Parameter: clean text, preserve structured_data
    """

    def __init__(self, settings=None) -> None:
        # settings param kept for API consistency; currently not needed
        pass

    # ------------------------------------------------------------------
    # Block-level cleaning
    # ------------------------------------------------------------------

    def clean_block(self, block: ContentBlock) -> ContentBlock:
        """Return a cleaned copy of *block* (original is never mutated)."""
        bt = block.block_type

        if bt == "code":
            cleaned_text = self._clean_code(block.text)
        elif bt == "function_signature":
            cleaned_text = self._clean_signature(block.text)
        elif bt in ("prose", "note", "example", "unknown"):
            cleaned_text = self._clean_prose(block.text)
        elif bt == "parameter_list":
            cleaned_text = self._clean_prose(block.text)
            # structured_data preserved below
        elif bt == "table":
            cleaned_text = self._clean_prose(block.text)
        else:
            cleaned_text = self._clean_prose(block.text)

        return ContentBlock(
            block_type=block.block_type,
            text=cleaned_text,
            heading=block.heading.strip().rstrip("#").strip(),
            language=block.language,
            structured_data=block.structured_data,
        )

    # ------------------------------------------------------------------
    # Document-level cleaning
    # ------------------------------------------------------------------

    def clean_document(self, doc: NormalizedDocument) -> NormalizedDocument:
        """Return a cleaned copy of *doc* (original is never mutated)."""
        # --- title ---
        title = self._clean_title(doc.title, doc.content_blocks, doc.url)

        # --- description ---
        description = self._clean_prose(doc.description) if doc.description else ""

        # --- content blocks ---
        cleaned_blocks = []
        for block in doc.content_blocks:
            cb = self.clean_block(block)
            if cb.text.strip():  # drop blocks that became empty
                cleaned_blocks.append(cb)

        return NormalizedDocument(
            doc_id=doc.doc_id,
            url=doc.url,
            title=title,
            description=description,
            content_blocks=cleaned_blocks,
            metadata=doc.metadata,
            template_id=doc.template_id,
            content_type=doc.content_type,
            source_link=doc.source_link,
            error=doc.error,
        )

    # ------------------------------------------------------------------
    # Private: per-type cleaning strategies
    # ------------------------------------------------------------------

    def _clean_code(self, text: str) -> str:
        """
        Minimal cleaning for code blocks.
        Preserves whitespace and indentation — only removes artifacts.
        """
        if not text:
            return text
        text = _RE_COPY_CLIPBOARD.sub("", text)
        text = _RE_PROGRESS_CHARS.sub("", text)
        # Strip trailing whitespace per line (but NOT leading — preserves indentation)
        lines = [line.rstrip() for line in text.splitlines()]
        # Remove standalone "Copy to clipboard" lines that may have survived
        lines = [l for l in lines if l.lower().strip() != "copy to clipboard"]
        return "\n".join(lines).strip()

    def _clean_signature(self, text: str) -> str:
        """
        Cleaning for function/method signature blocks.
        - Collapse multiple spaces to one
        - Remove [source] # markers
        - Truncate if inline parameter dump detected (> 500 chars)
        """
        if not text:
            return text
        text = _RE_SOURCE_MARKER.sub("", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()

        # Truncate at inline param/yields dump
        if len(text) > 500:
            for pattern in (_RE_PARAM_DUMP, _RE_YIELDS_DUMP):
                m = pattern.search(text)
                if m:
                    text = text[:m.start()].strip()
                    break

        return text

    def _clean_prose(self, text: str) -> str:
        """
        Full normalization pipeline for prose text.
        """
        if not text:
            return text
        text = clean_unicode(text)
        text = clean_whitespace(text)
        text = strip_boilerplate(text)
        # Remove lines that are only punctuation or whitespace
        lines = [
            line for line in text.splitlines()
            if not _RE_PUNCT_ONLY.match(line) or not line.strip()
        ]
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Private: title cleaning
    # ------------------------------------------------------------------

    def _clean_title(
        self,
        title: str,
        content_blocks: list[ContentBlock],
        url: str,
    ) -> str:
        """
        Clean a document title:
        - Strip trailing "#" characters
        - Collapse whitespace
        - Replace artifact titles (too long, HTML-heavy, repeated chunks)
        """
        if not title:
            return self._fallback_title(content_blocks, url)

        # Strip trailing hashes (e.g. "config_context #" → "config_context")
        title = _RE_TRAILING_HASH.sub("", title)
        title = title.strip().rstrip("#").strip()

        # Collapse inline whitespace
        title = re.sub(r"\s+", " ", title)

        if self._is_title_artifact(title):
            return self._fallback_title(content_blocks, url)

        # Hard truncate (after artifact check, so we don't truncate real titles)
        if len(title) > 200:
            title = title[:197] + "..."

        return title

    def _is_title_artifact(self, title: str) -> bool:
        """
        Return True if the title looks like a scraping artifact.

        Heuristics:
        - Length > 200 characters
        - Contains > 3 newlines (page body in title field)
        - Contains HTML tags
        - Contains repeated 50-char chunks (3+ times)
        """
        if len(title) > 200:
            return True
        if title.count("\n") > 3:
            return True
        if _RE_HTML_TAG.search(title):
            return True
        if _RE_TITLE_REPEATED.search(title):
            return True
        return False

    def _fallback_title(self, content_blocks: list[ContentBlock], url: str) -> str:
        """
        Derive a fallback title from the first block heading or URL path.
        """
        for block in content_blocks:
            if block.heading and len(block.heading) < 150:
                return block.heading.strip()
        # Last resort: last segment of URL path
        path = url.rstrip("/").rsplit("/", 1)[-1]
        return path.replace("-", " ").replace("_", " ").strip() or url
