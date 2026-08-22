"""
Tests for parent_child.py
"""

from rag.chunking.parent_child import ParentChildBuilder
from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument


class TestParentChildBuilder:
    def test_build_creates_parent_chunks(self):
        builder = ParentChildBuilder({"parent_max_tokens": 1500, "overlap_tokens": 75, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(doc_id="d1", url="http://test.com", title="Title", description="", content_blocks=[], metadata={}, template_id="", content_type="unknown")
        
        chunks = [
            Chunk(chunk_id=f"c{i}", doc_id="d1", url="http://test.com", content=f"text {i}", content_type="prose", heading_path=["Title", "Section"], chunk_index=i, token_count=10)
            for i in range(5)
        ]
        
        parents, children = builder.build(chunks, doc)
        assert len(parents) == 1
        assert len(children) == 5
        assert parents[0].metadata["is_parent"] is True
        assert len(parents[0].metadata["child_chunk_ids"]) == 5

    def test_children_reference_parent(self):
        builder = ParentChildBuilder({"parent_max_tokens": 1500, "overlap_tokens": 75, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(doc_id="d1", url="http://test.com", title="Title", description="", content_blocks=[], metadata={}, template_id="", content_type="unknown")
        
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", url="http://test.com", content="1", content_type="prose", heading_path=["Title", "S1"], chunk_index=0, token_count=10),
            Chunk(chunk_id="c2", doc_id="d1", url="http://test.com", content="2", content_type="prose", heading_path=["Title", "S1"], chunk_index=1, token_count=10)
        ]
        
        parents, children = builder.build(chunks, doc)
        parent_id = parents[0].chunk_id
        for child in children:
            assert child.parent_chunk_id == parent_id

    def test_parent_contains_child_content(self):
        builder = ParentChildBuilder({"parent_max_tokens": 1500, "overlap_tokens": 75, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(doc_id="d1", url="http://test.com", title="Title", description="", content_blocks=[], metadata={}, template_id="", content_type="unknown")
        
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", url="http://test.com", content="AAA", content_type="prose", heading_path=["Title"], chunk_index=0, token_count=10),
            Chunk(chunk_id="c2", doc_id="d1", url="http://test.com", content="BBB", content_type="prose", heading_path=["Title"], chunk_index=1, token_count=10)
        ]
        
        parents, children = builder.build(chunks, doc)
        assert "AAA" in parents[0].content
        assert "BBB" in parents[0].content

    def test_parent_ids_dont_collide_with_child_ids(self):
        builder = ParentChildBuilder({"parent_max_tokens": 1500, "overlap_tokens": 75, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(doc_id="d1", url="http://test.com", title="Title", description="", content_blocks=[], metadata={}, template_id="", content_type="unknown")
        chunks = [Chunk(chunk_id="c1", doc_id="d1", url="http://test.com", content="1", content_type="prose", heading_path=["Title"], chunk_index=0, token_count=10)]
        parents, children = builder.build(chunks, doc)
        
        # Parent ID should be different from any child ID and generated using offset
        assert parents[0].chunk_id != "c1"
        assert parents[0].chunk_index >= 10000

    def test_large_section_produces_multiple_parents(self):
        builder = ParentChildBuilder({"parent_max_tokens": 50, "overlap_tokens": 10, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(doc_id="d1", url="http://test.com", title="Title", description="", content_blocks=[], metadata={}, template_id="", content_type="unknown")
        
        # Generate enough text to exceed parent_max_tokens (50)
        chunks = [
            Chunk(chunk_id=f"c{i}", doc_id="d1", url="http://test.com", content="word " * 30, content_type="prose", heading_path=["Title"], chunk_index=i, token_count=30)
            for i in range(3)
        ]
        
        parents, children = builder.build(chunks, doc)
        assert len(parents) > 1
        
        # Children point to the first parent
        for child in children:
            assert child.parent_chunk_id == parents[0].chunk_id
