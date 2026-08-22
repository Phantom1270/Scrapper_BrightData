import pytest
import time
from rag.serving.cache import QueryCache
from rag.serving.schemas import QueryApiResponse


@pytest.fixture
def sample_result():
    return QueryApiResponse(
        answer="A", sources=[], citations=[], confidence="high",
        retrieval_time_ms=10.0, generation_time_ms=20.0, total_time_ms=30.0,
        llm_model="test"
    )


class TestQueryCache:
    def test_cache_set_and_get(self, sample_result):
        cache = QueryCache()
        cache.set("Q?", 5, None, None, True, sample_result)
        
        cached = cache.get("Q?", 5, None, None, True)
        assert cached is not None
        assert cached.answer == "A"

    def test_cache_miss(self):
        cache = QueryCache()
        assert cache.get("Q?", 5, None, None, True) is None

    def test_cache_ttl_expiration(self, sample_result):
        cache = QueryCache(ttl_seconds=0)
        cache.set("Q?", 5, None, None, True, sample_result)
        time.sleep(0.01) # ensure expiration
        assert cache.get("Q?", 5, None, None, True) is None

    def test_cache_lru_eviction(self, sample_result):
        cache = QueryCache(max_size=2)
        cache.set("Q1", 5, None, None, True, sample_result)
        cache.set("Q2", 5, None, None, True, sample_result)
        cache.set("Q3", 5, None, None, True, sample_result)
        
        # Q1 should be evicted
        assert cache.get("Q1", 5, None, None, True) is None
        assert cache.get("Q2", 5, None, None, True) is not None
        assert cache.get("Q3", 5, None, None, True) is not None

    def test_cache_clear(self, sample_result):
        cache = QueryCache()
        cache.set("Q1", 5, None, None, True, sample_result)
        cache.clear()
        assert cache.get("Q1", 5, None, None, True) is None

    def test_cache_stats(self, sample_result):
        cache = QueryCache()
        cache.set("Q1", 5, None, None, True, sample_result)
        
        cache.get("Q1", 5, None, None, True) # Hit
        cache.get("Q2", 5, None, None, True) # Miss
        
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1

    def test_cache_different_params_different_keys(self, sample_result):
        cache = QueryCache()
        cache.set("Q?", 5, None, None, True, sample_result)
        
        assert cache.get("Q?", 10, None, None, True) is None
        assert cache.get("Q?", 5, "api", None, True) is None

    def test_cache_hit_rate(self, sample_result):
        cache = QueryCache()
        cache.set("Q1", 5, None, None, True, sample_result)
        
        cache.get("Q1", 5, None, None, True)
        cache.get("Q1", 5, None, None, True)
        cache.get("Q1", 5, None, None, True)
        cache.get("Q2", 5, None, None, True)
        
        stats = cache.stats()
        assert stats["hit_rate"] == 0.75
