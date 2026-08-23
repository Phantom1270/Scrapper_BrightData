import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.serving.app import create_app


class TestApp:
    def test_create_app_returns_fastapi(self):
        app = create_app()
        assert isinstance(app, FastAPI)
        assert app.title == "Documentation RAG API"
        assert app.version == "1.0.0"

    def test_app_has_routes(self):
        app = create_app()
        # Flatten routes from included routers
        all_paths = set()
        for route in app.routes:
            if hasattr(route, 'path'):
                all_paths.add(route.path)
            if hasattr(route, 'routes'):
                for sub_route in route.routes:
                    if hasattr(sub_route, 'path'):
                        all_paths.add(sub_route.path)
            if hasattr(route, 'original_router') and hasattr(route, 'include_context'):
                prefix = getattr(route.include_context, 'prefix', '')
                for sub_route in getattr(route.original_router, 'routes', []):
                    if hasattr(sub_route, 'path'):
                        all_paths.add(prefix + sub_route.path)

        assert "/api/v1/query" in all_paths
        assert "/api/v1/index" in all_paths
        assert "/api/v1/health" in all_paths

    def test_app_docs_accessible(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/docs")
        assert response.status_code == 200

    def test_app_cors_headers(self):
        app = create_app()
        client = TestClient(app)
        
        # Test standard CORS preflight
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-Requested-With"
        }
        
        response = client.options("/api/v1/query", headers=headers)
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
