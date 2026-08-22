"""
Chunking strategy for Jupyter notebook documents.
"""

from typing import List

from rag.chunking.strategies.base import BaseChunkingStrategy
from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument


class NotebookChunkingStrategy(BaseChunkingStrategy):
    """Strategy for Jupyter notebooks mapping markdown and code cells to chunks."""

    def chunk(self, document: NormalizedDocument) -> List[Chunk]:
        chunks = []
        chunk_index = 0
        current_section_heading = ""
        
        i = 0
        while i < len(document.content_blocks):
            block = document.content_blocks[i]
            
            if block.block_type == "prose":
                if block.heading:
                    current_section_heading = block.heading
                
                # Check for contextual code pairing
                # If this prose block is very short and the next block is code, merge it into the code chunk
                tokens = self._count_tokens(block.text)
                if tokens < self.min_tokens and (i + 1 < len(document.content_blocks)) and document.content_blocks[i + 1].block_type == "code":
                    next_block = document.content_blocks[i + 1]
                    content = block.text + "\n\n" + next_block.text
                    heading_path = [document.title]
                    if current_section_heading:
                        heading_path.append(current_section_heading)
                    heading_path.append("Code")
                    
                    c = self._make_chunk(document, content, "code", heading_path, chunk_index, "code", next_block.language)
                    if c.token_count > self.max_tokens:
                        c.metadata["is_oversized"] = True
                        c.metadata["oversized_code"] = True
                    c.metadata["context_prefix"] = current_section_heading
                    
                    chunks.append(c)
                    chunk_index += 1
                    i += 2  # Skip the next code block since we merged it
                    continue

                # Normal prose handling
                heading_path = [document.title]
                if current_section_heading:
                    heading_path.append(current_section_heading)
                    
                texts = self._split_text_by_tokens(block.text, self.max_tokens, self.overlap_tokens, self.encoding_name)
                for text in texts:
                    c = self._make_chunk(document, text, "prose", heading_path, chunk_index, block.block_type)
                    chunks.append(c)
                    chunk_index += 1
                i += 1
                
            elif block.block_type == "code":
                heading_path = [document.title]
                if current_section_heading:
                    heading_path.append(current_section_heading)
                heading_path.append("Code")
                
                c = self._make_chunk(document, block.text, "code", heading_path, chunk_index, "code", block.language)
                if c.token_count > self.max_tokens:
                    c.metadata["is_oversized"] = True
                    c.metadata["oversized_code"] = True
                    
                # If it follows a prose block (which wasn't short enough to merge), 
                # we still add context_prefix
                if i > 0 and document.content_blocks[i-1].block_type == "prose":
                    c.metadata["preceding_heading"] = current_section_heading
                    
                chunks.append(c)
                chunk_index += 1
                i += 1
                
            else:
                # Default treat as prose
                heading_path = [document.title]
                if current_section_heading:
                    heading_path.append(current_section_heading)
                    
                texts = self._split_text_by_tokens(block.text, self.max_tokens, self.overlap_tokens, self.encoding_name)
                for text in texts:
                    c = self._make_chunk(document, text, "prose", heading_path, chunk_index, block.block_type)
                    chunks.append(c)
                    chunk_index += 1
                i += 1

        return chunks
