"""
Tests for generic.py
"""

from rag.chunking.strategies.generic import GenericChunkingStrategy
from rag.models.document import NormalizedDocument, ContentBlock


class TestGenericChunkingStrategy:
    def test_all_blocks_produce_chunks(self):
        strategy = GenericChunkingStrategy({"max_tokens": 512, "overlap_tokens": 75, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Title", description="",
            content_blocks=[
                ContentBlock(block_type="prose", text="P1", heading="H1"),
                ContentBlock(block_type="note", text="N1")
            ],
            metadata={}, template_id="", content_type="unknown"
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) == 2

    def test_code_is_atomic(self):
        strategy = GenericChunkingStrategy({"max_tokens": 10, "overlap_tokens": 5, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Title", description="",
            content_blocks=[
                ContentBlock(block_type="code", text="word " * 20, heading="H1"),
            ],
            metadata={}, template_id="", content_type="unknown"
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content_type == "code"
        assert chunks[0].metadata.get("is_oversized") is True

    def test_prose_is_split_by_tokens(self):
        strategy = GenericChunkingStrategy({"max_tokens": 10, "overlap_tokens": 5, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Title", description="",
            content_blocks=[
                ContentBlock(block_type="prose", text="word " * 20, heading="H1"),
            ],
            metadata={}, template_id="", content_type="unknown"
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.content_type == "prose"
            assert chunk.heading_path == ["Title", "H1"]
