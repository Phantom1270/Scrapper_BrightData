import pytest
from rag.llm.base import BaseLLMClient

class MockLLMClient(BaseLLMClient):
    def chat(self, messages, temperature=0.1, max_tokens=300):
        return f"Mock response to {len(messages)} messages"

    def get_model_name(self):
        return "mock-model"

    def is_available(self):
        return True


class TestBaseLLMClient:
    def test_chat_returns_string(self):
        client = MockLLMClient()
        response = client.chat([{"role": "user", "content": "hello"}])
        assert isinstance(response, str)
        assert "Mock response" in response

    def test_chat_passes_messages_through(self):
        client = MockLLMClient()
        response = client.chat([{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}])
        assert "2 messages" in response

    def test_get_model_name_returns_string(self):
        client = MockLLMClient()
        assert isinstance(client.get_model_name(), str)
        assert client.get_model_name() == "mock-model"

    def test_is_available_returns_bool(self):
        client = MockLLMClient()
        assert client.is_available() is True
