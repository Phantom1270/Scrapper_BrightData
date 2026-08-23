import pytest
from unittest.mock import MagicMock

from rag.generation.prompts import (
    PromptBuilder,
    SYSTEM_PROMPT,
    LOW_CONFIDENCE_PROMPT,
    NO_CONTEXT_PROMPT
)
from rag.models.retrieval import RetrievalResult


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.generation.max_context_tokens = 3000
    settings.chunking.encoding_name = "cl100k_base"
    return settings


class TestPromptBuilder:
    def test_build_messages_high_confidence(self, mock_settings):
        builder = PromptBuilder(mock_settings)
        messages = builder.build_messages(
            query="test query", 
            context="test context", 
            has_context=True, 
            confidence="high"
        )
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "technical documentation assistant" in messages[0]["content"]
        assert messages[0]["content"] == SYSTEM_PROMPT
        
        assert messages[1]["role"] == "user"
        assert "Context:\ntest context\n\n---\n\nQuestion: test query" in messages[1]["content"]

    def test_build_messages_low_confidence(self, mock_settings):
        builder = PromptBuilder(mock_settings)
        messages = builder.build_messages(
            query="test query", 
            context="test context", 
            has_context=True, 
            confidence="low"
        )
        
        assert messages[0]["content"] == LOW_CONFIDENCE_PROMPT
        assert "may not fully answer" in messages[0]["content"]
        assert "Question: test query" in messages[1]["content"]

    def test_build_messages_no_context(self, mock_settings):
        builder = PromptBuilder(mock_settings)
        messages = builder.build_messages(
            query="test query", 
            context="", 
            has_context=False, 
            confidence="none"
        )
        
        assert messages[0]["content"] == NO_CONTEXT_PROMPT
        assert "no relevant documentation was found" in messages[0]["content"]
        assert messages[1]["content"] == "Question: test query"
        assert "Context:" not in messages[1]["content"]

    def test_assess_confidence_empty_results(self, mock_settings):
        builder = PromptBuilder(mock_settings)
        assert builder.assess_confidence([]) == "none"

    def test_assess_confidence_low_scores(self, mock_settings):
        builder = PromptBuilder(mock_settings)
        # Vector score < 0.3
        results = [RetrievalResult("c1", "a", "u", [], "t", 0.2, "vector")]
        assert builder.assess_confidence(results) == "low"

    def test_assess_confidence_high_scores(self, mock_settings):
        builder = PromptBuilder(mock_settings)
        # Vector score > 0.5, multiple results
        results = [
            RetrievalResult("c1", "a", "u", [], "t", 0.8, "vector"),
            RetrievalResult("c2", "a", "u", [], "t", 0.7, "vector")
        ]
        assert builder.assess_confidence(results) == "high"

    def test_assess_confidence_few_low_results(self, mock_settings):
        builder = PromptBuilder(mock_settings)
        # Vector score < 0.5 and < 2 results
        results = [RetrievalResult("c1", "a", "u", [], "t", 0.4, "vector")]
        assert builder.assess_confidence(results) == "low"

    def test_assess_confidence_reranked_thresholds_low(self, mock_settings):
        builder = PromptBuilder(mock_settings)
        # Reranked score < 1.0
        results = [RetrievalResult("c1", "a", "u", [], "t", 0.5, "reranked")]
        assert builder.assess_confidence(results) == "low"

    def test_assess_confidence_reranked_thresholds_high(self, mock_settings):
        builder = PromptBuilder(mock_settings)
        # Reranked score >= 1.0
        results = [RetrievalResult("c1", "a", "u", [], "t", 1.5, "reranked")]
        assert builder.assess_confidence(results) == "high"
