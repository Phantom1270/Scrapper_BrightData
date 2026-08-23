"""
Context Builder for assembling retrieved chunks.
"""

from typing import List
import logging

from rag.models.retrieval import RetrievalResult
from rag.utils.tokens import count_tokens
from rag.utils.text import truncate_to_tokens

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Assembles retrieved chunks into a context string for the LLM."""

    def __init__(self, settings=None):
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()
            
        self.max_context_tokens = settings.generation.max_context_tokens
        self.encoding_name = settings.chunking.encoding_name

    def build_context(self, results: List[RetrievalResult]) -> str:
        """
        Assemble retrieved chunks into a single context string.
        """
        if not results:
            return ""

        # Sort results by score descending
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        context_parts = []
        total_tokens = 0
        
        for result in sorted_results:
            source_info = self._format_source(result)
            chunk_text = f"--- {source_info} ---\n{result.content}\n"
            
            chunk_tokens = count_tokens(chunk_text, self.encoding_name)
            
            if total_tokens + chunk_tokens > self.max_context_tokens:
                remaining_tokens = self.max_context_tokens - total_tokens
                if remaining_tokens > 50:
                    truncated = truncate_to_tokens(
                        chunk_text, remaining_tokens, self.encoding_name
                    )
                    context_parts.append(truncated)
                break
                
            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
            
        return "\n".join(context_parts)

    def _format_source(self, result: RetrievalResult) -> str:
        """Format a source identifier for a chunk."""
        if result.heading_path:
            return f"Source: {' > '.join(result.heading_path)}"
        elif result.url:
            return f"Source: {result.url}"
        else:
            return f"Source: Chunk {result.chunk_id}"

    def estimate_token_usage(self, results: List[RetrievalResult]) -> dict:
        """Estimate how many tokens would be used by the context."""
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        total_tokens = 0
        chunks_included = 0
        chunks_truncated = 0
        chunks_skipped = 0
        
        for result in sorted_results:
            source_info = self._format_source(result)
            chunk_text = f"--- {source_info} ---\n{result.content}\n"
            
            chunk_tokens = count_tokens(chunk_text, self.encoding_name)
            
            if total_tokens + chunk_tokens > self.max_context_tokens:
                remaining = self.max_context_tokens - total_tokens
                if remaining > 50:
                    chunks_truncated += 1
                    total_tokens += remaining
                else:
                    chunks_skipped += 1
            else:
                chunks_included += 1
                total_tokens += chunk_tokens
                
        # Remainder skipped
        chunks_skipped += len(sorted_results) - (chunks_included + chunks_truncated + chunks_skipped)
        
        return {
            "total_available": self.max_context_tokens,
            "total_used": total_tokens,
            "chunks_included": chunks_included,
            "chunks_truncated": chunks_truncated,
            "chunks_skipped": chunks_skipped,
        }
