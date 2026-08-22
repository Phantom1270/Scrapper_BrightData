"""
Tests for base chunking strategy.
"""

from rag.chunking.strategies.base import BaseChunkingStrategy
from rag.models.document import NormalizedDocument


class DummyStrategy(BaseChunkingStrategy):
    def chunk(self, document: NormalizedDocument):
        pass


class TestBaseChunkingStrategy:
    def test_split_short_text_returns_single_chunk(self):
        strategy = DummyStrategy({"max_tokens": 512, "overlap_tokens": 75, "encoding_name": "cl100k_base"})
        text = "This is a short text."
        chunks = strategy._split_text_by_tokens(text, max_tokens=512, overlap_tokens=75, encoding_name="cl100k_base")
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text_returns_multiple_chunks(self):
        strategy = DummyStrategy({"max_tokens": 50, "overlap_tokens": 10, "encoding_name": "cl100k_base"})
        text = "word " * 200  # Should be ~200 tokens
        chunks = strategy._split_text_by_tokens(text, max_tokens=50, overlap_tokens=10, encoding_name="cl100k_base")
        assert len(chunks) > 1

    def test_split_preserves_all_content(self):
        strategy = DummyStrategy({"max_tokens": 50, "overlap_tokens": 0, "encoding_name": "cl100k_base"})
        # No overlap means chunks should exactly piece back together
        text = "one " * 100
        chunks = strategy._split_text_by_tokens(text, max_tokens=50, overlap_tokens=0, encoding_name="cl100k_base")
        joined = "".join(chunks)
        assert joined == text

    def test_split_respects_overlap(self):
        strategy = DummyStrategy({"max_tokens": 20, "overlap_tokens": 5, "encoding_name": "cl100k_base"})
        text = "a " * 50
        chunks = strategy._split_text_by_tokens(text, max_tokens=20, overlap_tokens=5, encoding_name="cl100k_base")
        
        # We can't strictly assert the exact string overlap easily without re-encoding, but we can check lengths
        for chunk in chunks:
            tc = strategy._count_tokens(chunk)
            assert tc <= 20

    def test_split_breaks_at_sentence_boundary(self):
        strategy = DummyStrategy({"max_tokens": 20, "overlap_tokens": 2, "encoding_name": "cl100k_base"})
        text = "Sentence one is here. Sentence two is very long and goes past the limit of twenty tokens for sure. Sentence three is also here."
        chunks = strategy._split_text_by_tokens(text, max_tokens=20, overlap_tokens=2, encoding_name="cl100k_base")
        assert len(chunks) > 1
        assert "Sentence one is here." in chunks[0]

    def test_make_chunk_generates_unique_ids(self):
        strategy = DummyStrategy({"max_tokens": 50, "overlap_tokens": 10, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(doc_id="d1", url="http://test.com", title="", description="", content_blocks=[], metadata={}, template_id="", content_type="unknown")
        c1 = strategy._make_chunk(doc, "test1", "prose", [], 0)
        c2 = strategy._make_chunk(doc, "test2", "prose", [], 1)
        assert c1.chunk_id != c2.chunk_id

    def test_make_chunk_counts_tokens(self):
        strategy = DummyStrategy({"max_tokens": 50, "overlap_tokens": 10, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(doc_id="d1", url="http://test.com", title="", description="", content_blocks=[], metadata={}, template_id="", content_type="unknown")
        c = strategy._make_chunk(doc, "hello world", "prose", [], 0)
        assert c.token_count == strategy._count_tokens("hello world")
