import pytest
from rag.retrieval.query_transform.hyde import HyDETransformer
from rag.llm.base import BaseLLMClient

class MockLLMClient(BaseLLMClient):
    def __init__(self, response="This is a hypothetical document."):
        self._response = response
        self.error_to_raise = None
        
    def chat(self, messages, temperature=0.1, max_tokens=300):
        if self.error_to_raise:
            raise self.error_to_raise
        return self._response

    def get_model_name(self):
        return "mock"

    def is_available(self):
        return True


class TestHyDETransformer:
    def test_transform_returns_single_item(self):
        client = MockLLMClient()
        transformer = HyDETransformer(llm_client=client)
        
        result = transformer.transform("test query")
        
        assert isinstance(result, list)
        assert len(result) == 1

    def test_transform_returns_different_text_than_query(self):
        client = MockLLMClient("A generated hypothetical document.")
        transformer = HyDETransformer(llm_client=client)
        
        result = transformer.transform("test query")
        
        assert result[0] == "A generated hypothetical document."
        assert result[0] != "test query"

    def test_transform_handles_connection_error(self):
        client = MockLLMClient()
        client.error_to_raise = ConnectionError("Timeout")
        transformer = HyDETransformer(llm_client=client)
        
        result = transformer.transform("test query")
        
        assert len(result) == 1
        assert result[0] == "test query"

    def test_transform_handles_runtime_error(self):
        client = MockLLMClient()
        client.error_to_raise = RuntimeError("API error")
        transformer = HyDETransformer(llm_client=client)
        
        result = transformer.transform("test query")
        
        assert len(result) == 1
        assert result[0] == "test query"

    def test_get_transform_name(self):
        client = MockLLMClient()
        transformer = HyDETransformer(llm_client=client)
        assert transformer.get_transform_name() == "hyde"
