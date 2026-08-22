"""
Chunking strategy for tutorials, guides, and example pages.
"""

from typing import List

from rag.chunking.strategies.base import BaseChunkingStrategy
from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument


class TutorialChunkingStrategy(BaseChunkingStrategy):
    """Strategy for tutorials with prose sections and code blocks."""

    def chunk(self, document: NormalizedDocument) -> List[Chunk]:
        chunks = []
        chunk_index = 0

        for block in document.content_blocks:
            if block.block_type in ("code", "example"):
                # Atomic
                heading_path = [document.title, block.heading or "Code Example"]
                c = self._make_chunk(document, block.text, block.block_type, heading_path, chunk_index, block.block_type, block.language)
                if c.token_count > self.max_tokens:
                    c.metadata["is_oversized"] = True
                    c.metadata["oversized_code"] = True
                chunks.append(c)
                chunk_index += 1
                
            elif block.block_type == "note":
                heading_path = [document.title, block.heading or "Notes"]
                texts = self._split_text_by_tokens(block.text, self.max_tokens, self.overlap_tokens, self.encoding_name)
                for text in texts:
                    c = self._make_chunk(document, text, "note", heading_path, chunk_index, "note")
                    chunks.append(c)
                    chunk_index += 1
                    
            else:
                # Prose (including sections)
                heading_path = [document.title, block.heading] if block.heading else [document.title]
                texts = self._split_text_by_tokens(block.text, self.max_tokens, self.overlap_tokens, self.encoding_name)
                for text in texts:
                    c = self._make_chunk(document, text, "prose", heading_path, chunk_index, block.block_type)
                    chunks.append(c)
                    chunk_index += 1

        return chunks
