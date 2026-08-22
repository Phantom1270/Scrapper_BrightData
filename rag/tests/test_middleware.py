import pytest
import logging
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.serving.middleware import ErrorHandlerMiddleware, RequestLoggingMiddleware


@pytest.fixture
def error_app():
    app = FastAPI()
    app.add_middleware(ErrorHandlerMiddleware)
    
    @app.get("/value_error")
    async def value_error():
        raise ValueError("Bad value")
        
    @app.get("/not_found")
    async def not_found():
        raise FileNotFoundError("Missing file")
        
    @app.get("/conn_error")
    async def conn_error():
        raise ConnectionError("No connection")
        
    @app.get("/generic_error")
    async def generic_error():
        raise RuntimeError("Something broke")
        
    return app


class TestErrorHandlerMiddleware:
    def test_error_handler_value_error(self, error_app):
        client = TestClient(error_app)
        response = client.get("/value_error")
        assert response.status_code == 400
        assert response.json()["error"] == "Bad Request"
        assert response.json()["detail"] == "Bad value"

    def test_error_handler_file_not_found(self, error_app):
        client = TestClient(error_app)
        response = client.get("/not_found")
        assert response.status_code == 404
        assert response.json()["error"] == "Not Found"

    def test_error_handler_connection_error(self, error_app):
        client = TestClient(error_app)
        response = client.get("/conn_error")
        assert response.status_code == 503
        assert response.json()["error"] == "Service Unavailable"

    def test_error_handler_generic_exception(self, error_app):
        client = TestClient(error_app)
        response = client.get("/generic_error")
        assert response.status_code == 500
        assert response.json()["error"] == "Internal server error"


class TestRequestLoggingMiddleware:
    def test_request_logging(self, caplog):
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        
        @app.get("/test")
        async def test_route():
            return {"status": "ok"}
            
        client = TestClient(app)
        
        with caplog.at_level(logging.INFO):
            response = client.get("/test?param=1")
            
        assert response.status_code == 200
        
        # Check if log was recorded
        log_records = caplog.records
        assert len(log_records) > 0
        log_message = log_records[0].message
        assert "GET" in log_message
        assert "/test?param=1" in log_message
        assert "200" in log_message
        assert "ms" in log_message
