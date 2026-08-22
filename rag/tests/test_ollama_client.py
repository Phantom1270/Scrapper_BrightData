import pytest
import json
from unittest.mock import patch, MagicMock

from rag.llm.ollama_client import OllamaClient
from rag.config.settings import Settings


@pytest.fixture
def mock_settings():
    settings = Settings()
    settings.llm.provider = "ollama"
    settings.llm.model = "qwen2.5:3b"
    settings.llm.base_url = "http://localhost:11434"
    return settings


class TestOllamaClient:
    @patch.object(OllamaClient, "_make_request")
    def test_chat_sends_correct_request_body(self, mock_make_request, mock_settings):
        mock_make_request.return_value = {"message": {"content": "response"}}
        
        client = OllamaClient(mock_settings)
        messages = [{"role": "user", "content": "hello"}]
        
        client.chat(messages, temperature=0.5, max_tokens=100)
        
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args
        
        assert args[0] == "POST"
        assert args[1] == "http://localhost:11434/api/chat"
        
        body = kwargs.get("body")
        assert body is not None
        assert body["model"] == "qwen2.5:3b"
        assert body["messages"] == messages
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0.5
        assert body["options"]["num_predict"] == 100

    @patch.object(OllamaClient, "_make_request")
    def test_chat_returns_content_string(self, mock_make_request, mock_settings):
        mock_make_request.return_value = {"message": {"content": "Expected response text"}}
        
        client = OllamaClient(mock_settings)
        response = client.chat([{"role": "user", "content": "hello"}])
        
        assert response == "Expected response text"

    @patch.object(OllamaClient, "_make_request")
    def test_chat_raises_connection_error_on_refused(self, mock_make_request, mock_settings):
        mock_make_request.side_effect = ConnectionError("Connection refused")
        
        client = OllamaClient(mock_settings)
        with pytest.raises(ConnectionError, match="Connection refused"):
            client.chat([{"role": "user", "content": "hello"}])

    @patch.object(OllamaClient, "_make_request")
    def test_chat_raises_runtime_error_on_http_error(self, mock_make_request, mock_settings):
        mock_make_request.side_effect = RuntimeError("HTTP error from Ollama")
        
        client = OllamaClient(mock_settings)
        with pytest.raises(RuntimeError, match="HTTP error from Ollama"):
            client.chat([{"role": "user", "content": "hello"}])

    @patch.object(OllamaClient, "_make_request")
    def test_is_available_returns_true_when_running(self, mock_make_request, mock_settings):
        mock_make_request.return_value = {"tags": []}
        
        client = OllamaClient(mock_settings)
        assert client.is_available() is True

    @patch.object(OllamaClient, "_make_request")
    def test_is_available_returns_false_when_unreachable(self, mock_make_request, mock_settings):
        mock_make_request.side_effect = ConnectionError("Timeout")
        
        client = OllamaClient(mock_settings)
        assert client.is_available() is False

    def test_get_model_name(self, mock_settings):
        client = OllamaClient(mock_settings)
        assert client.get_model_name() == "qwen2.5:3b"

    def test_default_settings_loaded(self):
        client = OllamaClient()
        assert client.model_name == "qwen2.5:3b"
        assert client.base_url == "http://localhost:11434"
