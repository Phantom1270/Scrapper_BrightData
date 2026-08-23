import pytest
from unittest.mock import MagicMock
from rag.models.query import QueryRequest
from rag.models.retrieval import RetrievalResult
from rag.retrieval.retrieval_engine import RetrievalEngine


@pytest.fixture
def mock_search_engine():
    engine = MagicMock()
    # search_both returns (vector_results, bm25_results)
    engine.search_both.return_value = (
        [RetrievalResult("c1", "content1", "url1", [], "tutorial", 0.9, "vector")],
        [RetrievalResult("c2", "content2", "url2", [], "api_reference", 2.0, "bm25")]
    )
    return engine


@pytest.fixture
def mock_reranker():
    reranker = MagicMock()
    reranker.get_model_name.return_value = "mock-reranker"
    def mock_rerank(query, candidates, top_k):
        for c in candidates:
            c.source = "reranked"
        return candidates[:top_k]
    reranker.rerank.side_effect = mock_rerank
    return reranker


@pytest.fixture
def mock_transformer():
    transformer = MagicMock()
    transformer.transform.return_value = ["original query", "variation query"]
    transformer.get_transform_name.return_value = "mock_transform"
    return transformer


@pytest.fixture
def mock_filter_builder():
    builder = MagicMock()
    builder.build_filters.return_value = {"content_type": None}
    builder.should_filter.return_value = False
    return builder


@pytest.fixture
def engine(mock_search_engine, mock_filter_builder):
    settings = MagicMock()
    settings.retrieval.top_k = 5
    settings.retrieval.candidate_k = 20
    settings.retrieval.vector_weight = 0.6
    settings.retrieval.bm25_weight = 0.4
    settings.retrieval.use_query_transform = False
    settings.reranker.enabled = False
    settings.retrieval.rrf_k = 60
    
    return RetrievalEngine(
        settings=settings,
        search_engine=mock_search_engine,
        fusion=None, # will use default
        filter_builder=mock_filter_builder
    )


class TestRetrievalEngine:
    def test_retrieve_basic_hybrid(self, engine):
        results = engine.retrieve("test query")
        
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, RetrievalResult)
            assert r.source == "hybrid"

    def test_retrieve_with_reranker(self, engine, mock_reranker):
        engine.set_reranker(mock_reranker)
        
        results = engine.retrieve("test query")
        
        assert len(results) == 2
        for r in results:
            assert r.source == "reranked"
        
        mock_reranker.rerank.assert_called_once()

    def test_retrieve_with_query_transformer(self, engine, mock_transformer, mock_search_engine):
        engine.set_query_transformer(mock_transformer)
        engine.use_query_transform = True
        
        results = engine.retrieve("test query")
        
        # search_both should be called for original + variation = 2 times
        assert mock_search_engine.search_both.call_count == 2
        mock_transformer.transform.assert_called_once_with("test query")

    def test_query_transform_disabled_by_config(self, engine, mock_transformer, mock_search_engine):
        engine.set_query_transformer(mock_transformer)
        engine.use_query_transform = False
        
        results = engine.retrieve("test query")
        
        # Only called once for original query
        assert mock_search_engine.search_both.call_count == 1
        mock_transformer.transform.assert_not_called()

    def test_retrieve_with_filter(self, engine, mock_filter_builder):
        mock_filter_builder.build_filters.return_value = {"content_type": "api_reference"}
        
        results = engine.retrieve("test query")
        
        # Only c2 should remain because it has content_type="api_reference"
        assert len(results) == 1
        assert results[0].chunk_id == "c2"

    def test_retrieve_respects_top_k(self, engine):
        engine.top_k = 1
        results = engine.retrieve("test query")
        assert len(results) == 1

    def test_retrieve_respects_request_object(self, engine):
        request = QueryRequest(question="test query", top_k=1)
        results = engine.retrieve(request=request)
        assert len(results) == 1

    def test_retrieve_request_overrides_string(self, engine, mock_filter_builder):
        request = QueryRequest(question="override query")
        engine.retrieve(query="original query", request=request)
        
        mock_filter_builder.build_filters.assert_called_once_with("override query")

    def test_retrieve_empty_query_raises(self, engine):
        with pytest.raises(ValueError, match="Query string cannot be empty"):
            engine.retrieve("")
            
        with pytest.raises(ValueError, match="Query string cannot be empty"):
            engine.retrieve(None)

    def test_retrieve_and_format_returns_dict(self, engine):
        formatted = engine.retrieve_and_format("test query")
        
        assert isinstance(formatted, dict)
        assert "query" in formatted
        assert "chunks" in formatted
        assert "filters_applied" in formatted
        assert "transform_used" in formatted
        assert "reranker_used" in formatted

    def test_retrieve_and_format_chunks_have_all_fields(self, engine):
        formatted = engine.retrieve_and_format("test query")
        
        chunks = formatted["chunks"]
        assert len(chunks) == 2
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "content" in chunk
            assert "heading" in chunk
            assert "url" in chunk
            assert "score" in chunk
            assert "source" in chunk
            assert "content_type" in chunk

    def test_retrieve_graceful_when_transformer_fails(self, engine, mock_transformer, mock_search_engine):
        mock_transformer.transform.side_effect = RuntimeError("Transformation failed")
        engine.set_query_transformer(mock_transformer)
        engine.use_query_transform = True
        
        results = engine.retrieve("test query")
        
        # Should gracefully fallback to original query -> 1 search_both call
        assert mock_search_engine.search_both.call_count == 1
        assert len(results) > 0

    def test_retrieve_graceful_when_reranker_fails(self, engine, mock_reranker):
        mock_reranker.rerank.side_effect = RuntimeError("Reranking failed")
        engine.set_reranker(mock_reranker)
        
        results = engine.retrieve("test query")
        
        # Should gracefully return hybrid results
        assert len(results) > 0
        for r in results:
            assert r.source == "hybrid"

    def test_retrieve_deduplicates_multi_query_results(self, engine, mock_search_engine):
        # Even if search_both is called twice and returns same chunks, final results should be deduplicated
        engine.search_engine.search_both.return_value = (
            [RetrievalResult("c1", "content1", "", [], "", 0.9, "vector")],
            []
        )
        
        mock_transformer = MagicMock()
        mock_transformer.transform.return_value = ["q1", "q2", "q3"]
        engine.set_query_transformer(mock_transformer)
        engine.use_query_transform = True
        
        results = engine.retrieve("test query")
        
        assert mock_search_engine.search_both.call_count == 3
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_set_query_transformer_after_construction(self, engine):
        assert engine.query_transformer is None
        engine.set_query_transformer(MagicMock())
        assert engine.query_transformer is not None

    def test_set_reranker_after_construction(self, engine):
        assert engine.reranker is None
        engine.set_reranker(MagicMock())
        assert engine.reranker is not None

    def test_retrieve_no_reranker_flag(self, engine, mock_reranker):
        engine.set_reranker(mock_reranker)
        
        request = QueryRequest(question="test", use_reranking=False)
        results = engine.retrieve(request=request)
        
        mock_reranker.rerank.assert_not_called()
        for r in results:
            assert r.source == "hybrid"
