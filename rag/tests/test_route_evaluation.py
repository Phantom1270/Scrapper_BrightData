import pytest
import os
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from rag.serving.routes.evaluation import router
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


class TestRouteEvaluation:
    @patch('rag.serving.routes.evaluation.TestSet')
    @patch('rag.serving.routes.evaluation.EvaluationRunner')
    @patch('rag.serving.routes.evaluation.ReportGenerator')
    @patch('rag.serving.routes.evaluation.get_generation_engine')
    @patch('rag.serving.routes.evaluation.get_settings')
    @patch('os.path.exists')
    def test_run_evaluation_success(
        self, mock_exists, mock_get_settings, mock_get_engine,
        MockReportGenerator, MockEvaluationRunner, MockTestSet, client
    ):
        mock_exists.return_value = True
        
        mock_settings = MagicMock()
        mock_settings.evaluation.test_set_path = "test.json"
        mock_get_settings.return_value = mock_settings
        
        mock_test_set = MagicMock()
        MockTestSet.load.return_value = mock_test_set
        
        mock_runner = MagicMock()
        mock_report = MagicMock()
        mock_report.total_questions = 10
        mock_report.aggregate_metrics = {"precision": 1.0}
        mock_runner.evaluate_test_set.return_value = mock_report
        MockEvaluationRunner.return_value = mock_runner
        
        mock_report_gen = MagicMock()
        mock_report_gen.generate_report.return_value = "# Report"
        mock_report_gen.save_report.return_value = "/path/to/report.md"
        MockReportGenerator.return_value = mock_report_gen
        
        response = client.post(
            "/evaluation/run",
            json={} # Empty json to test default path handling
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["total_questions"] == 10
        assert data["aggregate_metrics"] == {"precision": 1.0}
        assert data["report_path"] == "/path/to/report.md"

    @patch('rag.serving.routes.evaluation.get_settings')
    @patch('os.path.exists')
    def test_run_evaluation_file_not_found(self, mock_exists, mock_get_settings, client):
        mock_exists.return_value = False
        
        mock_settings = MagicMock()
        mock_settings.evaluation.test_set_path = "missing.json"
        mock_get_settings.return_value = mock_settings
        
        response = client.post("/evaluation/run", json={})
        
        assert response.status_code == 404

    @patch('rag.serving.routes.evaluation.get_settings')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=MagicMock)
    def test_get_report_success(self, mock_open, mock_exists, mock_get_settings, client):
        mock_exists.return_value = True
        
        mock_settings = MagicMock()
        mock_settings.evaluation.report_path = "report.md"
        mock_get_settings.return_value = mock_settings
        
        # Setup mock file read
        mock_file = MagicMock()
        mock_file.read.return_value = "# Test Report Markdown"
        mock_open.return_value.__enter__.return_value = mock_file
        
        response = client.get("/evaluation/report")
        
        assert response.status_code == 200
        assert response.text == "# Test Report Markdown"
        assert response.headers["content-type"] == "text/markdown; charset=utf-8"

    @patch('rag.serving.routes.evaluation.get_settings')
    @patch('os.path.exists')
    def test_get_report_not_found(self, mock_exists, mock_get_settings, client):
        mock_exists.return_value = False
        
        mock_settings = MagicMock()
        mock_settings.evaluation.report_path = "missing_report.md"
        mock_get_settings.return_value = mock_settings
        
        response = client.get("/evaluation/report")
        
        assert response.status_code == 404
