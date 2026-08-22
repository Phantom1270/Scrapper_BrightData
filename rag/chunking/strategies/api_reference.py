"""
Chunking strategy for API reference documentation.
"""

from typing import List

from rag.chunking.strategies.base import BaseChunkingStrategy
from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument


class ApiReferenceChunkingStrategy(BaseChunkingStrategy):
    """Strategy for API reference documentation with parameters and signatures."""

    def chunk(self, document: NormalizedDocument) -> List[Chunk]:
        chunks = []
        chunk_index = 0
        
        i = 0
        while i < len(document.content_blocks):
            block = document.content_blocks[i]
            
            if block.block_type == "function_signature":
                # Atomic
                heading_path = [document.title, "Signature"]
                c = self._make_chunk(document, block.text, "function_signature", heading_path, chunk_index, "function_signature", block.language)
                chunks.append(c)
                chunk_index += 1
                i += 1
                
            elif block.block_type == "parameter_list":
                # Check for parameter grouping optimization
                # Group consecutive parameter_list blocks that are short
                param_group = []
                current_group_tokens = 0
                j = i
                while j < len(document.content_blocks) and document.content_blocks[j].block_type == "parameter_list":
                    next_block = document.content_blocks[j]
                    tokens = self._count_tokens(next_block.text)
                    if tokens >= self.min_tokens and len(param_group) == 0:
                        # If a single param is already large, it shouldn't be grouped with others
                        # if we're just starting a group. We'll handle it individually.
                        pass
                    
                    if len(param_group) > 0 and current_group_tokens + tokens > self.max_tokens:
                        break # Exceeds max_tokens for a group
                        
                    param_group.append((next_block, tokens))
                    current_group_tokens += tokens
                    j += 1
                    
                    # If this single parameter was large, break after adding it (handled individually below)
                    if len(param_group) == 1 and tokens >= self.min_tokens:
                        break

                if len(param_group) > 1:
                    # We grouped multiple parameters
                    combined_text = "\n\n".join(b.text for b, _ in param_group)
                    heading_path = [document.title, "Parameters"]
                    c = self._make_chunk(document, combined_text, "parameter_list", heading_path, chunk_index, "parameter_list")
                    chunks.append(c)
                    chunk_index += 1
                    i = j
                else:
                    # Single parameter
                    p_block, p_tokens = param_group[0]
                    heading_path = [document.title, p_block.heading] if p_block.heading else [document.title, "Parameter"]
                    
                    if p_tokens <= self.max_tokens:
                        c = self._make_chunk(document, p_block.text, "parameter_list", heading_path, chunk_index, "parameter_list")
                        chunks.append(c)
                        chunk_index += 1
                    else:
                        # Exceeds max_tokens, need to split. 
                        # We try to keep the prefix (e.g. "Parameter: name\nType: type_info") in all sub-chunks.
                        # For simplicity in this base implementation, we split normally. 
                        # To keep prefix, we could split by newline and extract it, but standard split handles it functionally well enough
                        texts = self._split_text_by_tokens(p_block.text, self.max_tokens, self.overlap_tokens, self.encoding_name)
                        for text in texts:
                            c = self._make_chunk(document, text, "parameter_list", heading_path, chunk_index, "parameter_list")
                            chunks.append(c)
                            chunk_index += 1
                    i += 1
                    
            elif block.block_type in ("code", "example"):
                # Atomic
                heading_path = [document.title, block.heading or "Code Example"]
                c = self._make_chunk(document, block.text, block.block_type, heading_path, chunk_index, block.block_type, block.language)
                if c.token_count > self.max_tokens:
                    c.metadata["is_oversized"] = True
                    c.metadata["oversized_code"] = True
                chunks.append(c)
                chunk_index += 1
                i += 1
                
            elif block.block_type == "note":
                heading_path = [document.title, block.heading or "Notes"]
                texts = self._split_text_by_tokens(block.text, self.max_tokens, self.overlap_tokens, self.encoding_name)
                for text in texts:
                    c = self._make_chunk(document, text, "note", heading_path, chunk_index, "note")
                    chunks.append(c)
                    chunk_index += 1
                i += 1
                
            else:
                # Prose and others
                heading_path = [document.title, block.heading] if block.heading else [document.title]
                texts = self._split_text_by_tokens(block.text, self.max_tokens, self.overlap_tokens, self.encoding_name)
                for text in texts:
                    c = self._make_chunk(document, text, "prose", heading_path, chunk_index, block.block_type)
                    chunks.append(c)
                    chunk_index += 1
                i += 1

        return chunks
