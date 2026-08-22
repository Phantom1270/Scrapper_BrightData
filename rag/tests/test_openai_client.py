import pytest
from unittest.mock import patch, MagicMock

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
class TestOpenAIClient:
    def test_chat_returns_string(self):
        from rag.llm.openai_client import OpenAIClient
        
        with patch("rag.llm.openai_client.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Expected response"
            mock_client.chat.completions.create.return_value = mock_response
            
            client = OpenAIClient(api_key="fake-key", model="gpt-4o-mini")
            response = client.chat([{"role": "user", "content": "hello"}])
            
            assert response == "Expected response"
            mock_client.chat.completions.create.assert_called_once()

    def test_get_model_name(self):
        from rag.llm.openai_client import OpenAIClient
        
        with patch("rag.llm.openai_client.OpenAI"):
            client = OpenAIClient(api_key="fake-key", model="gpt-4o")
            assert client.get_model_name() == "gpt-4o"

    def test_is_available_returns_false_without_key(self):
        from rag.llm.openai_client import OpenAIClient
        from openai import OpenAIError
        
        with patch("rag.llm.openai_client.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.models.list.side_effect = OpenAIError("Auth failed")
            
            client = OpenAIClient(api_key="fake-key")
            assert client.is_available() is False
