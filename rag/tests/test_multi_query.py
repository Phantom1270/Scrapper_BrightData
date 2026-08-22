import pytest
from unittest.mock import MagicMock

from rag.retrieval.query_transform.multi_query import MultiQueryTransformer
from rag.llm.base import BaseLLMClient

class MockLLMClient(BaseLLMClient):
    def __init__(self, response="variation 1\nvariation 2\nvariation 3"):
        self._response = response
        self.error_to_raise = None
        
    def chat(self, messages, temperature=0.7, max_tokens=200):
        if self.error_to_raise:
            raise self.error_to_raise
        return self._response

    def get_model_name(self):
        return "mock"

    def is_available(self):
        return True


class TestMultiQueryTransformer:
    def test_transform_returns_original_plus_variations(self):
        client = MockLLMClient("how to set config\nconfiguring things\nsettings guide")
        transformer = MultiQueryTransformer(llm_client=client, n_queries=3)
        
        result = transformer.transform("how to configure")
        
        assert len(result) == 4
        assert result[0] == "how to configure"
        assert result[1] == "how to set config"
        assert result[2] == "configuring things"
        assert result[3] == "settings guide"

    def test_transform_returns_at_least_original_on_empty_response(self):
        client = MockLLMClient("")
        transformer = MultiQueryTransformer(llm_client=client)
        
        result = transformer.transform("how to configure")
        
        assert len(result) == 1
        assert result[0] == "how to configure"

    def test_transform_handles_connection_error(self):
        client = MockLLMClient()
        client.error_to_raise = ConnectionError("Timeout")
        transformer = MultiQueryTransformer(llm_client=client)
        
        result = transformer.transform("test query")
        
        assert len(result) == 1
        assert result[0] == "test query"

    def test_transform_handles_runtime_error(self):
        client = MockLLMClient()
        client.error_to_raise = RuntimeError("API error")
        transformer = MultiQueryTransformer(llm_client=client)
        
        result = transformer.transform("test query")
        
        assert len(result) == 1
        assert result[0] == "test query"

    def test_transform_respects_n_queries(self):
        client = MockLLMClient("v1\nv2\nv3\nv4\nv5")
        transformer = MultiQueryTransformer(llm_client=client, n_queries=2)
        
        result = transformer.transform("test")
        
        assert len(result) == 3
        assert result[0] == "test"
        assert result[1] == "v1"
        assert result[2] == "v2"

    def test_get_transform_name(self):
        client = MockLLMClient()
        transformer = MultiQueryTransformer(llm_client=client)
        assert transformer.get_transform_name() == "multi_query"
