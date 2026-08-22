import pytest
from pydantic import ValidationError
from rag.serving.schemas import (
    QueryApiRequest, QueryApiResponse, IndexRequest, HealthResponse
)


class TestSchemas:
    def test_query_api_request_valid(self):
        req = QueryApiRequest(question="Q?", top_k=10, use_reranking=False)
        assert req.question == "Q?"
        assert req.top_k == 10
        assert req.use_reranking is False
        assert req.filter_content_type is None

    def test_query_api_request_empty_question_fails(self):
        with pytest.raises(ValidationError):
            QueryApiRequest(question="")

    def test_query_api_request_long_question_fails(self):
        with pytest.raises(ValidationError):
            QueryApiRequest(question="a" * 2001)

    def test_query_api_request_defaults(self):
        req = QueryApiRequest(question="Q?")
        assert req.top_k == 5
        assert req.use_reranking is True
        assert req.filter_content_type is None
        assert req.filter_doc_id is None

    def test_query_api_response_creation(self):
        resp = QueryApiResponse(
            answer="A", sources=[], citations=[], confidence="high",
            retrieval_time_ms=10.0, generation_time_ms=20.0, total_time_ms=30.0,
            llm_model="test", cached=True
        )
        assert resp.cached is True

    def test_index_request_defaults(self):
        req = IndexRequest()
        assert req.force_rebuild is False

    def test_health_response_creation(self):
        resp = HealthResponse(
            status="healthy", version="1.0", uptime_seconds=10.5, components={}
        )
        assert resp.status == "healthy"
