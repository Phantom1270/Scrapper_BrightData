"""
Builds parent-child chunk relationships for advanced retrieval.
"""

from typing import List, Tuple
from collections import defaultdict

from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument
from rag.chunking.strategies.generic import GenericChunkingStrategy
from rag.utils.ids import generate_chunk_id


class ParentChildBuilder:
    """Builds hierarchical chunks for context-aware retrieval."""

    def __init__(self, settings=None, config=None):
        if config is not None:
            settings = config
            
        if isinstance(settings, dict):
            self.config = dict(settings)
            if "parent_max_tokens" not in self.config:
                self.config["parent_max_tokens"] = 1500
        else:
            if settings is None:
                from rag.config.settings import get_settings
                settings = get_settings()
            self.config = {
                "max_tokens": settings.chunking.max_tokens,
                "overlap_tokens": settings.chunking.overlap_tokens,
                "encoding_name": settings.chunking.encoding_name,
                "parent_max_tokens": getattr(settings.chunking, "parent_max_tokens", 1500)
            }
                
        # We use a dummy strategy just to access the split_text_by_tokens utility
        self._strategy = GenericChunkingStrategy(self.config)
        self.parent_max_tokens = self.config["parent_max_tokens"]

    def build(self, chunks: List[Chunk], document: NormalizedDocument) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Groups chunks into larger parent chunks.
        Returns (parent_chunks, child_chunks).
        """
        if not chunks:
            return [], []

        # 1. Group chunks by their section-level heading
        groups = defaultdict(list)
        for chunk in chunks:
            # heading_path[0] is document title, heading_path[1] is section level (if exists)
            if len(chunk.heading_path) > 1:
                group_key = tuple(chunk.heading_path[:2])
            else:
                group_key = tuple(chunk.heading_path)
            groups[group_key].append(chunk)

        parent_chunks = []
        parent_index = 0
        
        # 2. Process each group
        for group_key, child_group in groups.items():
            combined_content = "\n\n".join(c.content for c in child_group)
            
            # Split if combined content is too large
            parent_texts = self._strategy._split_text_by_tokens(
                combined_content, 
                max_tokens=self.parent_max_tokens, 
                overlap_tokens=self.config.get("overlap_tokens", 75),
                encoding_name=self.config.get("encoding_name", "cl100k_base")
            )
            
            group_heading_path = list(group_key)
            
            # 3. Create parent chunks
            group_parents = []
            for text in parent_texts:
                p_chunk_id = generate_chunk_id(document.doc_id, 10000 + parent_index)
                
                parent = Chunk(
                    chunk_id=p_chunk_id,
                    doc_id=document.doc_id,
                    url=document.url,
                    content=text,
                    content_type="prose",
                    heading_path=group_heading_path,
                    chunk_index=10000 + parent_index,
                    token_count=self._strategy._count_tokens(text),
                    block_type="",
                    language="",
                    metadata={
                        "is_parent": True,
                        "child_chunk_ids": [c.chunk_id for c in child_group]
                    }
                )
                parent_chunks.append(parent)
                group_parents.append(parent)
                parent_index += 1
                
            # 4. Link children to the first parent of their group
            # (If a group was split, we just link to the first parent for simplicity,
            # or we could link to the specific parent that contains their text,
            # but usually parent_max_tokens is large enough that splits are rare)
            for child in child_group:
                child.parent_chunk_id = group_parents[0].chunk_id

        return parent_chunks, chunks
