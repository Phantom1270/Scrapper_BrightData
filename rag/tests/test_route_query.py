import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from rag.serving.routes.query import router
from rag.serving.middleware import ErrorHandlerMiddleware
from rag.generation.generator import GenerationResult
from rag.serving.schemas import QueryApiResponse


@pytest.fixture
def app():
    app = FastAPI()
    app.add_middleware(ErrorHandlerMiddleware)
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_cache():
    cache = MagicMock()
    cache.get.return_value = None
    return cache


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.generate.return_value = GenerationResult(
        answer="Mock answer",
        sources=[{"id": "c1", "url": "url1"}],
        citations=[{"source_id": 1, "text": "Mock"}],
        confidence="high",
        retrieval_time_ms=10.0,
        generation_time_ms=20.0,
        chunks_retrieved=5,
        chunks_used_in_context=2,
        llm_model="mock-model",
        transform_used=None,
        reranker_used="mock-reranker"
    )
    return engine


class TestRouteQuery:
    @patch('rag.serving.routes.query.get_cache')
    @patch('rag.serving.routes.query.get_generation_engine')
    def test_query_endpoint_success(self, mock_get_engine, mock_get_cache, client, mock_cache, mock_engine):
        mock_get_cache.return_value = mock_cache
        mock_get_engine.return_value = mock_engine
        
        response = client.post(
            "/query",
            json={"question": "What is it?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Mock answer"
        assert data["cached"] is False
        assert data["total_time_ms"] == 30.0
        
        # Verify cache.set was called
        mock_cache.set.assert_called_once()

    def test_query_endpoint_empty_question(self, client):
        response = client.post(
            "/query",
            json={"question": ""}
        )
        assert response.status_code == 422 # Validation error

    @patch('rag.serving.routes.query.get_cache')
    def test_query_endpoint_caches_result(self, mock_get_cache, client):
        mock_cache = MagicMock()
        mock_cache.get.return_value = QueryApiResponse(
            answer="Cached answer", sources=[], citations=[], confidence="high",
            retrieval_time_ms=5.0, generation_time_ms=5.0, total_time_ms=10.0,
            cached=False, llm_model="test"
        )
        mock_get_cache.return_value = mock_cache
        
        response = client.post(
            "/query",
            json={"question": "What is it?"}
        )
        
        assert response.status_code == 200
        assert response.json()["answer"] == "Cached answer"
        assert response.json()["cached"] is True

    @patch('rag.serving.routes.query.get_cache')
    @patch('rag.serving.routes.query.get_generation_engine')
    def test_query_endpoint_with_filters(self, mock_get_engine, mock_get_cache, client, mock_cache, mock_engine):
        mock_get_cache.return_value = mock_cache
        mock_get_engine.return_value = mock_engine
        
        response = client.post(
            "/query",
            json={"question": "Q?", "filter_content_type": "api_reference"}
        )
        
        assert response.status_code == 200
        
        # Verify the engine was called with correct QueryRequest
        call_kwargs = mock_engine.generate.call_args.kwargs
        qr = call_kwargs['request']
        assert qr.filter_content_type == "api_reference"

    @patch('rag.serving.routes.query.get_cache')
    @patch('rag.serving.routes.query.get_generation_engine')
    def test_query_endpoint_engine_error(self, mock_get_engine, mock_get_cache, client, mock_cache):
        mock_get_cache.return_value = mock_cache
        mock_engine = MagicMock()
        mock_engine.generate.side_effect = RuntimeError("Engine broke")
        mock_get_engine.return_value = mock_engine
        
        response = client.post(
            "/query",
            json={"question": "Q?"}
        )
        
        assert response.status_code == 500
        assert response.json()["detail"] == "Engine broke"

    @patch('rag.serving.routes.query.get_cache')
    @patch('rag.serving.routes.query.get_generation_engine')
    def test_query_endpoint_connection_error(self, mock_get_engine, mock_get_cache, client, mock_cache):
        mock_get_cache.return_value = mock_cache
        mock_engine = MagicMock()
        mock_engine.generate.side_effect = ConnectionError("Ollama down")
        mock_get_engine.return_value = mock_engine
        
        response = client.post(
            "/query",
            json={"question": "Q?"}
        )
        
        assert response.status_code == 503
        assert response.json()["error"] == "Service Unavailable"
