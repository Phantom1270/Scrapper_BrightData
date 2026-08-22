import pytest
from unittest.mock import MagicMock, patch

from rag.generation.generator import GenerationEngine, GenerationResult
from rag.models.query import QueryRequest, QueryResponse
from rag.models.retrieval import RetrievalResult
from rag.generation.prompts import LOW_CONFIDENCE_PROMPT, NO_CONTEXT_PROMPT


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.generation.temperature = 0.5
    settings.generation.max_tokens = 500
    settings.generation.max_context_tokens = 3000
    settings.chunking.encoding_name = "cl100k_base"
    return settings


@pytest.fixture
def mock_retrieval_engine():
    engine = MagicMock()
    results = [
        RetrievalResult("c1", "content 1", "url1", ["H1"], "t", 0.9, "vector"),
        RetrievalResult("c2", "content 2", "url2", ["H2"], "t", 0.8, "vector"),
        RetrievalResult("c3", "content 3", "url3", ["H3"], "t", 0.7, "vector"),
    ]
    engine.retrieve.return_value = results
    engine.query_transformer = MagicMock()
    engine.query_transformer.get_transform_name.return_value = "mock_transform"
    engine.use_query_transform = True
    engine.reranker = MagicMock()
    engine.reranker.get_model_name.return_value = "mock_reranker"
    return engine


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat.return_value = "Generated answer [Source: H1]."
    client.get_model_name.return_value = "mock_model"
    return client


class TestGenerationEngine:
    def test_generate_basic(self, mock_settings, mock_retrieval_engine, mock_llm_client):
        engine = GenerationEngine(
            settings=mock_settings,
            retrieval_engine=mock_retrieval_engine,
            llm_client=mock_llm_client
        )
        
        result = engine.generate("test question")
        
        assert isinstance(result, GenerationResult)
        assert result.answer == "Generated answer [Source: H1]."
        assert len(result.sources) == 3
        assert len(result.citations) == 1
        assert result.citations[0]["matched"] is True
        assert result.citations[0]["chunk_id"] == "c1"
        assert result.confidence == "high"
        assert result.llm_model == "mock_model"

    def test_generate_no_results(self, mock_settings, mock_retrieval_engine, mock_llm_client):
        mock_retrieval_engine.retrieve.return_value = []
        mock_llm_client.chat.return_value = "I don't know."
        
        engine = GenerationEngine(
            settings=mock_settings,
            retrieval_engine=mock_retrieval_engine,
            llm_client=mock_llm_client
        )
        
        result = engine.generate("test question")
        
        assert result.confidence == "none"
        assert result.chunks_retrieved == 0
        assert result.chunks_used_in_context == 0
        
        # Check that it used NO_CONTEXT_PROMPT
        mock_llm_client.chat.assert_called_once()
        messages = mock_llm_client.chat.call_args[1]["messages"]
        assert messages[0]["content"] == NO_CONTEXT_PROMPT

    def test_generate_low_confidence(self, mock_settings, mock_retrieval_engine, mock_llm_client):
        mock_retrieval_engine.retrieve.return_value = [
            RetrievalResult("c1", "content", "url", [], "t", 0.1, "vector")
        ]
        
        engine = GenerationEngine(
            settings=mock_settings,
            retrieval_engine=mock_retrieval_engine,
            llm_client=mock_llm_client
        )
        
        result = engine.generate("test question")
        
        assert result.confidence == "low"
        
        mock_llm_client.chat.assert_called_once()
        messages = mock_llm_client.chat.call_args[1]["messages"]
        assert messages[0]["content"] == LOW_CONFIDENCE_PROMPT

    def test_generate_respects_request_object(self, mock_settings, mock_retrieval_engine, mock_llm_client):
        engine = GenerationEngine(
            settings=mock_settings,
            retrieval_engine=mock_retrieval_engine,
            llm_client=mock_llm_client
        )
        
        request = QueryRequest(question="test question from request")
        engine.generate(request=request)
        
        mock_retrieval_engine.retrieve.assert_called_once_with(query=None, request=request)

    def test_generate_empty_query_raises(self, mock_settings, mock_retrieval_engine, mock_llm_client):
        engine = GenerationEngine(
            settings=mock_settings,
            retrieval_engine=mock_retrieval_engine,
            llm_client=mock_llm_client
        )
        
        with pytest.raises(ValueError, match="Query string cannot be empty"):
            engine.generate("")

    def test_generate_llm_error_propagates(self, mock_settings, mock_retrieval_engine, mock_llm_client):
        mock_llm_client.chat.side_effect = RuntimeError("API failed")
        
        engine = GenerationEngine(
            settings=mock_settings,
            retrieval_engine=mock_retrieval_engine,
            llm_client=mock_llm_client
        )
        
        with pytest.raises(RuntimeError, match="API failed"):
            engine.generate("test question")

    def test_generate_and_format_returns_query_response(self, mock_settings, mock_retrieval_engine, mock_llm_client):
        engine = GenerationEngine(
            settings=mock_settings,
            retrieval_engine=mock_retrieval_engine,
            llm_client=mock_llm_client
        )
        
        response = engine.generate_and_format("test question")
        
        assert isinstance(response, QueryResponse)
        assert response.answer == "Generated answer [Source: H1]."
        assert len(response.sources) == 3
        assert response.retrieval_time_ms >= 0
        assert response.generation_time_ms >= 0
        assert response.total_time_ms == response.retrieval_time_ms + response.generation_time_ms

    def test_generate_records_timing(self, mock_settings, mock_retrieval_engine, mock_llm_client):
        engine = GenerationEngine(
            settings=mock_settings,
            retrieval_engine=mock_retrieval_engine,
            llm_client=mock_llm_client
        )
        
        result = engine.generate("test question")
        
        assert result.retrieval_time_ms >= 0
        assert result.generation_time_ms >= 0

    def test_generate_sources_limited_to_5(self, mock_settings, mock_retrieval_engine, mock_llm_client):
        mock_retrieval_engine.retrieve.return_value = [
            RetrievalResult(f"c{i}", f"content {i}", f"url{i}", [], "t", 0.9, "vector")
            for i in range(10)
        ]
        
        engine = GenerationEngine(
            settings=mock_settings,
            retrieval_engine=mock_retrieval_engine,
            llm_client=mock_llm_client
        )
        
        result = engine.generate("test question")
        
        assert len(result.sources) == 5

    def test_generate_stream_falls_back_to_non_stream(self, mock_settings, mock_retrieval_engine, mock_llm_client, capsys):
        # Force AttributeError for chat_stream
        del mock_llm_client.chat_stream
        
        engine = GenerationEngine(
            settings=mock_settings,
            retrieval_engine=mock_retrieval_engine,
            llm_client=mock_llm_client
        )
        
        # Call with stream=True but mock client doesn't have chat_stream
        result = engine.generate("test question", stream=True)
        
        assert result.answer == "Generated answer [Source: H1]."
        mock_llm_client.chat.assert_called_once()
        
        # Verify it printed the fallback
        captured = capsys.readouterr()
        assert "Generated answer" in captured.out

    def test_generate_populates_chunks_retrieved(self, mock_settings, mock_retrieval_engine, mock_llm_client):
        engine = GenerationEngine(
            settings=mock_settings,
            retrieval_engine=mock_retrieval_engine,
            llm_client=mock_llm_client
        )
        
        result = engine.generate("test question")
        
        assert result.chunks_retrieved == 3
        # In mock, they are short, so all 3 should fit
        assert result.chunks_used_in_context == 3
