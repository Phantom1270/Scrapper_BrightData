import pytest
from typing import List
from rag.retrieval.query_transform.base import BaseQueryTransformer

class MockQueryTransformer(BaseQueryTransformer):
    def transform(self, query: str) -> List[str]:
        return [query, f"{query} variant"]

    def get_transform_name(self) -> str:
        return "mock_transform"

class TestBaseQueryTransformer:
    def test_transform_returns_list(self):
        transformer = MockQueryTransformer()
        result = transformer.transform("test query")
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_transform_includes_original(self):
        transformer = MockQueryTransformer()
        result = transformer.transform("test query")
        assert result[0] == "test query"

    def test_transform_name_returns_string(self):
        transformer = MockQueryTransformer()
        name = transformer.get_transform_name()
        assert isinstance(name, str)
        assert len(name) > 0
