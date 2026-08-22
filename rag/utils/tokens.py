"""
Token counting utilities.

Uses tiktoken when available; falls back to character-based approximation.
The encoder instance is cached to avoid reloading on every call.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Encoder cache
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _get_encoder(encoding_name: str):
    """
    Return a cached tiktoken encoder for the given encoding name.

    Raises ImportError if tiktoken is not installed — callers should
    catch this and fall back to approximation.
    """
    import tiktoken
    return tiktoken.get_encoding(encoding_name)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in text using the specified encoding.

    Falls back to a character-based approximation (~4 chars per token)
    if tiktoken is not available.
    """
    if not text:
        return 0
    try:
        enc = _get_encoder(encoding_name)
        return len(enc.encode(text))
    except ImportError:
        logger.warning(
            "tiktoken not available — using character approximation for token count."
        )
        return max(1, len(text) // 4)
    except Exception as exc:
        logger.warning("Token count failed (%s) — using approximation.", exc)
        return max(1, len(text) // 4)


def count_tokens_batch(
    texts: List[str],
    encoding_name: str = "cl100k_base",
) -> List[int]:
    """
    Count tokens for a batch of texts.

    More efficient than calling count_tokens in a loop because
    the encoder is reused across all texts in the batch.
    """
    if not texts:
        return []
    try:
        enc = _get_encoder(encoding_name)
        return [len(enc.encode(t)) if t else 0 for t in texts]
    except ImportError:
        logger.warning(
            "tiktoken not available — using character approximation for batch token count."
        )
        return [max(1, len(t) // 4) if t else 0 for t in texts]
    except Exception as exc:
        logger.warning("Batch token count failed (%s) — using approximation.", exc)
        return [max(1, len(t) // 4) if t else 0 for t in texts]
