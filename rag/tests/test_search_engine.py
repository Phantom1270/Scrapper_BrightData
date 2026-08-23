"""
Tests for SearchEngine.
"""

import pytest
from unittest.mock import Mock
from rag.search.search_engine import SearchEngine
from rag.search.bm25.bm25_index import BM25Index


class MockEmbedder:
    def embed_query(self, query):
        return [0.1] * 8


class MockVectorStore:
    def __init__(self):
        self.data = {
            "c1": {"id": "c1", "document": "sklearn.config_context is here", "score": 0.9, "metadata": {"url": "url1", "content_type": "prose"}},
            "c2": {"id": "c2", "document": "Another document", "score": 0.5, "metadata": {"url": "url2", "content_type": "code"}},
        }
        
    def search(self, query_embedding, top_k, content_type_filter=None, doc_id_filter=None):
        res = []
        for v in self.data.values():
            if content_type_filter and v["metadata"].get("content_type") != content_type_filter:
                continue
            res.append(v)
        return res[:top_k]
        
    def get_chunk_by_id(self, chunk_id):
        return self.data.get(chunk_id)


class MockStore:
    def get_chunk(self, chunk_id):
        return None


@pytest.fixture
def setup_search_engine():
    settings = Mock()
    settings.retrieval.top_k = 2
    settings.retrieval.candidate_k = 5
    
    store = MockStore()
    embedder = MockEmbedder()
    vector_store = MockVectorStore()
    bm25_index = BM25Index()
    bm25_index.build(
        ["c1", "c2", "c3", "c4", "c5"],
        ["sklearn.config_context is here", "Another document", "pad1", "pad2", "pad3"]
    )
    
    engine = SearchEngine(
        settings=settings,
        store=store,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index
    )
    return engine


class TestSearchEngine:
    def test_vector_search_returns_results(self, setup_search_engine):
        engine = setup_search_engine
        results = engine.vector_search("test query")
        assert len(results) == 2
        
    def test_vector_search_source_is_vector(self, setup_search_engine):
        engine = setup_search_engine
        results = engine.vector_search("test query")
        for r in results:
            assert r.source == "vector"

    def test_bm25_search_returns_results(self, setup_search_engine):
        engine = setup_search_engine
        results = engine.bm25_search("document")
        assert len(results) == 1
        assert results[0].chunk_id == "c2"
        
    def test_bm25_search_source_is_bm25(self, setup_search_engine):
        engine = setup_search_engine
        results = engine.bm25_search("document")
        for r in results:
            assert r.source == "bm25"

    def test_search_both_returns_two_lists(self, setup_search_engine):
        engine = setup_search_engine
        v_res, b_res = engine.search_both("document")
        assert isinstance(v_res, list)
        assert isinstance(b_res, list)
        assert len(v_res) == 2
        assert len(b_res) == 1

    def test_vector_search_with_content_type_filter(self, setup_search_engine):
        engine = setup_search_engine
        results = engine.vector_search("test query", content_type_filter="code")
        assert len(results) == 1
        assert results[0].content_type == "code"

    def test_vector_search_respects_top_k(self, setup_search_engine):
        engine = setup_search_engine
        results = engine.vector_search("test query", top_k=1)
        assert len(results) == 1

    def test_bm25_search_exact_function_name(self, setup_search_engine):
        engine = setup_search_engine
        results = engine.bm25_search("config_context")
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_search_empty_indexes(self, setup_search_engine):
        # Clear indexes
        setup_search_engine.vector_store.data = {}
        setup_search_engine.bm25_index.clear()
        
        v_res, b_res = setup_search_engine.search_both("test")
        assert len(v_res) == 0
        assert len(b_res) == 0

    def test_chunk_content_populated_in_results(self, setup_search_engine):
        engine = setup_search_engine
        results = engine.vector_search("test")
        assert len(results) > 0
        assert results[0].content != ""
        
        results2 = engine.bm25_search("document")
        assert len(results2) > 0
        assert results2[0].content != ""
