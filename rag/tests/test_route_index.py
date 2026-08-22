import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from rag.serving.routes.index import router
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


class TestRouteIndex:
    @patch('rag.serving.routes.index.get_index_builder')
    def test_trigger_index_success(self, mock_get_builder, client):
        mock_builder = MagicMock()
        mock_result = MagicMock()
        mock_result.total_chunks = 100
        mock_result.chunks_embedded = 100
        mock_result.embedding_model = "test"
        mock_result.embedding_dimension = 384
        mock_result.vector_store_count = 100
        mock_result.bm25_count = 100
        mock_result.skipped = 0
        mock_builder.build_all.return_value = mock_result
        mock_get_builder.return_value = mock_builder
        
        response = client.post(
            "/index",
            json={"force_rebuild": False}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["total_chunks"] == 100
        
        mock_builder.build_all.assert_called_once_with(force_rebuild=False)

    @patch('rag.serving.routes.index.get_index_builder')
    def test_trigger_index_force_rebuild(self, mock_get_builder, client):
        mock_builder = MagicMock()
        mock_builder.build_all.return_value = MagicMock()
        mock_get_builder.return_value = mock_builder
        
        response = client.post(
            "/index",
            json={"force_rebuild": True}
        )
        
        assert response.status_code == 200
        mock_builder.build_all.assert_called_once_with(force_rebuild=True)

    @patch('rag.serving.routes.index.get_index_builder')
    @patch('rag.serving.routes.index.get_store')
    def test_index_status_ready(self, mock_get_store, mock_get_builder, client):
        mock_store = MagicMock()
        mock_store.count_documents.return_value = 10
        mock_store.count_chunks.return_value = 100
        mock_get_store.return_value = mock_store
        
        mock_builder = MagicMock()
        mock_builder.vector_store.count.return_value = 100
        mock_builder.bm25_index.count.return_value = 100
        mock_get_builder.return_value = mock_builder
        
        response = client.get("/index/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["total_documents"] == 10
        assert data["total_chunks"] == 100

    @patch('rag.serving.routes.index.get_index_builder')
    @patch('rag.serving.routes.index.get_store')
    def test_index_status_empty(self, mock_get_store, mock_get_builder, client):
        mock_store = MagicMock()
        mock_store.count_documents.return_value = 0
        mock_store.count_chunks.return_value = 0
        mock_get_store.return_value = mock_store
        
        mock_builder = MagicMock()
        mock_builder.vector_store.count.return_value = 0
        mock_builder.bm25_index.count.return_value = 0
        mock_get_builder.return_value = mock_builder
        
        response = client.get("/index/status")
        
        assert response.status_code == 200
        assert response.json()["status"] == "empty"

    @patch('rag.serving.routes.index.get_index_builder')
    def test_trigger_index_error(self, mock_get_builder, client):
        mock_builder = MagicMock()
        mock_builder.build_all.side_effect = RuntimeError("Failed to build")
        mock_get_builder.return_value = mock_builder
        
        response = client.post("/index", json={})
        
        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to build"
