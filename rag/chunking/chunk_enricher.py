"""
Adds computed metadata to chunks after the initial splitting.
"""

from typing import List

from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument
from rag.chunking.heading_builder import HeadingBuilder


class ChunkEnricher:
    """Enriches chunk metadata with document context and computed stats."""

    def __init__(self, settings=None, config=None):
        if config is not None:
            settings = config
            
        self.heading_builder = HeadingBuilder()
        if isinstance(settings, dict):
            self.max_tokens = settings.get("max_tokens", 512)
        else:
            if settings is None:
                from rag.config.settings import get_settings
                settings = get_settings()
            self.max_tokens = settings.chunking.max_tokens

    def enrich(self, chunks: List[Chunk], document: NormalizedDocument) -> List[Chunk]:
        """
        Enrich a list of chunks with metadata. Mutates chunks in place and returns them.
        """
        for chunk in chunks:
            # Re-initialize metadata if it's somehow None, though dataclass defaults to dict
            if chunk.metadata is None:
                chunk.metadata = {}

            heading_text = self.heading_builder.path_to_string(chunk.heading_path)
            
            chunk.metadata["heading_text"] = heading_text
            chunk.metadata["word_count"] = len(chunk.content.split())
            chunk.metadata["char_count"] = len(chunk.content)
            
            # Add is_oversized flag based on token_count vs max_tokens
            if "is_oversized" not in chunk.metadata:
                chunk.metadata["is_oversized"] = chunk.token_count > self.max_tokens
            elif chunk.token_count > self.max_tokens:
                chunk.metadata["is_oversized"] = True
                
            source_section = chunk.heading_path[-1] if chunk.heading_path else ""
            chunk.metadata["source_section"] = source_section
            
            chunk.metadata["document_title"] = document.title
            chunk.metadata["document_url"] = document.url
            chunk.metadata["document_content_type"] = document.content_type
            chunk.metadata["template_id"] = document.template_id

        return chunks
