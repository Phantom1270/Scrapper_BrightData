"""
Generic fallback chunking strategy.
"""

from typing import List

from rag.chunking.strategies.base import BaseChunkingStrategy
from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument


class GenericChunkingStrategy(BaseChunkingStrategy):
    """Fallback strategy for documents with unknown content_type."""

    def chunk(self, document: NormalizedDocument) -> List[Chunk]:
        chunks = []
        chunk_index = 0

        for block in document.content_blocks:
            heading_path = [document.title]
            if block.heading:
                heading_path.append(block.heading)

            if block.block_type == "code":
                # Atomic
                c = self._make_chunk(
                    doc=document,
                    content=block.text,
                    content_type="code",
                    heading_path=heading_path,
                    chunk_index=chunk_index,
                    block_type="code",
                    language=block.language
                )
                if c.token_count > self.max_tokens:
                    c.metadata["is_oversized"] = True
                chunks.append(c)
                chunk_index += 1
            else:
                # Split by tokens
                texts = self._split_text_by_tokens(
                    block.text,
                    max_tokens=self.max_tokens,
                    overlap_tokens=self.overlap_tokens,
                    encoding_name=self.encoding_name
                )
                for text in texts:
                    c = self._make_chunk(
                        doc=document,
                        content=text,
                        content_type="prose",
                        heading_path=heading_path,
                        chunk_index=chunk_index,
                        block_type=block.block_type
                    )
                    chunks.append(c)
                    chunk_index += 1

        return chunks
