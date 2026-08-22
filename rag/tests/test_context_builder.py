import pytest
from unittest.mock import MagicMock

from rag.generation.context_builder import ContextBuilder
from rag.models.retrieval import RetrievalResult


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.generation.max_context_tokens = 3000
    settings.chunking.encoding_name = "cl100k_base"
    return settings


@pytest.fixture
def five_results():
    return [
        RetrievalResult("c1", "content one " * 10, "url1", ["H1", "H2"], "t", 0.9, "vector"),
        RetrievalResult("c2", "content two " * 10, "url2", [], "t", 0.8, "vector"),
        RetrievalResult("c3", "content three " * 10, "", [], "t", 0.7, "vector"),
        RetrievalResult("c4", "content four " * 10, "url4", ["H3"], "t", 0.6, "vector"),
        RetrievalResult("c5", "content five " * 10, "url5", [], "t", 0.5, "vector"),
    ]


class TestContextBuilder:
    def test_build_context_empty_results(self, mock_settings):
        builder = ContextBuilder(mock_settings)
        assert builder.build_context([]) == ""

    def test_build_context_includes_all_chunks(self, mock_settings, five_results):
        builder = ContextBuilder(mock_settings)
        context = builder.build_context(five_results)
        
        assert "content one" in context
        assert "content two" in context
        assert "content three" in context
        assert "content four" in context
        assert "content five" in context

    def test_build_context_respects_token_limit(self, mock_settings, five_results):
        mock_settings.generation.max_context_tokens = 50
        builder = ContextBuilder(mock_settings)
        
        context = builder.build_context(five_results)
        
        # c1 has ~20 tokens of content plus formatting, so it should fit, maybe partly c2
        assert "content one" in context
        # It definitely shouldn't fit all of them
        assert "content five" not in context

    def test_build_context_highest_score_first(self, mock_settings, five_results):
        # Mess up the order
        results = [five_results[4], five_results[2], five_results[0]]
        builder = ContextBuilder(mock_settings)
        
        context = builder.build_context(results)
        
        # c1 (score 0.9) should appear before c5 (score 0.5)
        pos1 = context.find("content one")
        pos5 = context.find("content five")
        
        assert pos1 != -1
        assert pos5 != -1
        assert pos1 < pos5

    def test_build_context_includes_source_info(self, mock_settings, five_results):
        builder = ContextBuilder(mock_settings)
        context = builder.build_context(five_results)
        
        assert "Source: H1 > H2" in context
        assert "Source: url2" in context
        assert "Source: Chunk c3" in context

    def test_build_context_truncates_large_chunk(self, mock_settings):
        mock_settings.generation.max_context_tokens = 100
        builder = ContextBuilder(mock_settings)
        
        large_content = "word " * 200
        results = [RetrievalResult("c1", large_content, "url", [], "t", 1.0, "vector")]
        
        context = builder.build_context(results)
        
        # It should contain some content but not all
        assert len(context) > 50
        assert len(context) < len(large_content) + 50

    def test_build_context_skips_tiny_remaining_budget(self, mock_settings):
        # 100 token limit
        mock_settings.generation.max_context_tokens = 100
        builder = ContextBuilder(mock_settings)
        
        results = [
            RetrievalResult("c1", "word " * 80, "url1", [], "t", 0.9, "vector"),
            RetrievalResult("c2", "word " * 80, "url2", [], "t", 0.8, "vector")
        ]
        
        context = builder.build_context(results)
        
        assert "url1" in context
        # Remaining budget is < 50 tokens, so c2 should be skipped entirely
        assert "url2" not in context

    def test_estimate_token_usage(self, mock_settings, five_results):
        mock_settings.generation.max_context_tokens = 3000
        builder = ContextBuilder(mock_settings)
        
        estimate = builder.estimate_token_usage(five_results)
        
        assert estimate["total_available"] == 3000
        assert estimate["chunks_included"] == 5
        assert estimate["chunks_truncated"] == 0
        assert estimate["chunks_skipped"] == 0
        assert estimate["total_used"] > 0
