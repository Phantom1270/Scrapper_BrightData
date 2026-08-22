import pytest
from typing import List
from rag.retrieval.reranker.base import BaseReranker
from rag.models.retrieval import RetrievalResult

class MockReranker(BaseReranker):
    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        if not candidates:
            return []
            
        # Score by string overlap for mock
        query_words = set(query.lower().split())
        for c in candidates:
            content_words = set(c.content.lower().split())
            c.score = float(len(query_words.intersection(content_words)))
            c.source = "reranked"
            
        sorted_c = sorted(candidates, key=lambda x: x.score, reverse=True)
        return sorted_c[:top_k]

    def get_model_name(self) -> str:
        return "mock-reranker"

@pytest.fixture
def candidates():
    return [
        RetrievalResult("c1", "the quick brown fox", "", [], "", 0.1, "vector"),
        RetrievalResult("c2", "jumps over the lazy dog", "", [], "", 0.2, "bm25"),
        RetrievalResult("c3", "the fox jumps", "", [], "", 0.3, "vector"),
        RetrievalResult("c4", "hello world", "", [], "", 0.4, "bm25"),
        RetrievalResult("c5", "a completely unrelated sentence", "", [], "", 0.5, "vector"),
        RetrievalResult("c6", "another unrelated one", "", [], "", 0.6, "vector")
    ]

class TestBaseReranker:
    def test_rerank_returns_top_k(self, candidates):
        reranker = MockReranker()
        results = reranker.rerank("fox jumps", candidates, top_k=2)
        assert len(results) == 2

    def test_rerank_updates_score(self, candidates):
        reranker = MockReranker()
        results = reranker.rerank("fox jumps", candidates, top_k=6)
        
        # c3 has "fox jumps", so intersection is 2
        assert results[0].chunk_id == "c3"
        assert results[0].score == 2.0
        
        # c1 has "fox", intersection 1
        # c2 has "jumps", intersection 1

    def test_rerank_updates_source_to_reranked(self, candidates):
        reranker = MockReranker()
        results = reranker.rerank("fox jumps", candidates, top_k=6)
        for r in results:
            assert r.source == "reranked"

    def test_rerank_sorted_by_score(self, candidates):
        reranker = MockReranker()
        results = reranker.rerank("fox jumps", candidates, top_k=6)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty_candidates(self):
        reranker = MockReranker()
        results = reranker.rerank("fox jumps", [], top_k=5)
        assert results == []

    def test_rerank_single_candidate(self, candidates):
        reranker = MockReranker()
        single = [candidates[0]]
        results = reranker.rerank("fox", single, top_k=5)
        assert len(results) == 1
        assert results[0].score == 1.0
