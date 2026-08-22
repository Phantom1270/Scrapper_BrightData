import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from rag.serving.routes.health import router
from rag.serving.middleware import ErrorHandlerMiddleware


@pytest.fixture
def app():
    app = FastAPI()
    app.add_middleware(ErrorHandlerMiddleware)
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestRouteHealth:
    @patch('rag.serving.routes.health.get_store')
    @patch('rag.serving.routes.health.get_index_builder')
    @patch('rag.llm.create_llm_client')
    @patch('rag.config.settings.get_settings')
    @patch('rag.serving.routes.health.get_start_time')
    def test_health_all_healthy(
        self, mock_get_time, mock_get_settings, mock_create_llm,
        mock_get_builder, mock_get_store, client
    ):
        mock_get_time.return_value = 1000.0
        
        mock_store = MagicMock()
        mock_store.count_documents.return_value = 10
        mock_get_store.return_value = mock_store
        
        mock_builder = MagicMock()
        mock_builder.vector_store.count.return_value = 50
        mock_builder.bm25_index.count.return_value = 50
        mock_get_builder.return_value = mock_builder
        
        mock_settings = MagicMock()
        mock_settings.llm.provider = "ollama"
        mock_get_settings.return_value = mock_settings
        
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.get_model_name.return_value = "test-model"
        mock_create_llm.return_value = mock_llm
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["uptime_seconds"] >= 0
        
        components = data["components"]
        assert components["storage"]["status"] == "healthy"
        assert components["llm"]["status"] == "healthy"
        assert components["vector_store"]["status"] == "healthy"
        assert components["bm25"]["status"] == "healthy"

    @patch('rag.serving.routes.health.get_store')
    @patch('rag.serving.routes.health.get_index_builder')
    @patch('rag.llm.create_llm_client')
    @patch('rag.config.settings.get_settings')
    def test_health_llm_unavailable(
        self, mock_get_settings, mock_create_llm,
        mock_get_builder, mock_get_store, client
    ):
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_builder = MagicMock()
        mock_get_builder.return_value = mock_builder
        
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = False # Degraded!
        mock_create_llm.return_value = mock_llm
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["llm"]["status"] == "unavailable"

    @patch('rag.serving.routes.health.get_store')
    @patch('rag.serving.routes.health.get_index_builder')
    @patch('rag.llm.create_llm_client')
    @patch('rag.config.settings.get_settings')
    def test_health_storage_error(
        self, mock_get_settings, mock_create_llm,
        mock_get_builder, mock_get_store, client
    ):
        mock_store = MagicMock()
        mock_store.count_documents.side_effect = RuntimeError("DB error")
        mock_get_store.return_value = mock_store
        
        mock_builder = MagicMock()
        mock_get_builder.return_value = mock_builder
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_create_llm.return_value = mock_llm
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["components"]["storage"]["status"] == "unhealthy"
        assert "DB error" in data["components"]["storage"]["error"]

    @patch('rag.serving.routes.health.health_check')
    def test_health_components_endpoint(self, mock_health_check, client):
        mock_health_check.return_value = MagicMock(components={"test": "ok"})
        
        response = client.get("/health/components")
        
        assert response.status_code == 200
        assert response.json() == {"test": "ok"}
