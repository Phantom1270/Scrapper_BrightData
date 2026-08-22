"""Utils package for the RAG system."""

from rag.utils.text import clean_whitespace, strip_boilerplate, clean_unicode, truncate_to_tokens, extract_code_language
from rag.utils.tokens import count_tokens, count_tokens_batch
from rag.utils.ids import generate_doc_id, generate_chunk_id
from rag.utils.hashing import content_hash, near_duplicate_ratio

__all__ = [
    "clean_whitespace",
    "strip_boilerplate",
    "clean_unicode",
    "truncate_to_tokens",
    "extract_code_language",
    "count_tokens",
    "count_tokens_batch",
    "generate_doc_id",
    "generate_chunk_id",
    "content_hash",
    "near_duplicate_ratio",
]
