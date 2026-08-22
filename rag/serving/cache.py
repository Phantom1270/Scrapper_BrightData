"""
In-memory query cache with TTL expiration.
"""

import time
import hashlib
from typing import Optional
from rag.serving.schemas import QueryApiResponse


class QueryCache:
    """Simple in-memory query cache with TTL expiration and LRU eviction."""

    def __init__(self, settings=None, ttl_seconds: int = None, max_size: int = 1000):
        if isinstance(settings, int):
            ttl_seconds = settings
            settings = None
            
        if ttl_seconds is None:
            if settings is None:
                from rag.config.settings import get_settings
                settings = get_settings()
            ttl_seconds = settings.serving.cache_ttl_seconds
            
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache = {}  # key -> (value, timestamp)
        self._access_order = []  # for LRU eviction
        self.hits = 0
        self.misses = 0

    def _make_key(self, question: str, top_k: int, content_type: Optional[str], doc_id: Optional[str], use_reranking: bool) -> str:
        """Create a deterministic cache key from query parameters."""
        ct = content_type or ""
        di = doc_id or ""
        key_str = f"{question}|{top_k}|{ct}|{di}|{use_reranking}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _evict_expired(self) -> None:
        """Remove all expired entries."""
        now = time.time()
        expired_keys = [k for k, (v, ts) in self._cache.items() if now - ts > self.ttl]
        for k in expired_keys:
            del self._cache[k]
            if k in self._access_order:
                self._access_order.remove(k)

    def get(self, question: str, top_k: int, content_type: Optional[str], doc_id: Optional[str], use_reranking: bool) -> Optional[QueryApiResponse]:
        """Look up a cached result."""
        self._evict_expired()
        
        key = self._make_key(question, top_k, content_type, doc_id, use_reranking)
        
        if key in self._cache:
            value, ts = self._cache[key]
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            self.hits += 1
            return value
            
        self.misses += 1
        return None

    def set(self, question: str, top_k: int, content_type: Optional[str], doc_id: Optional[str], use_reranking: bool, result: QueryApiResponse) -> None:
        """Store a result in the cache."""
        self._evict_expired()
        
        key = self._make_key(question, top_k, content_type, doc_id, use_reranking)
        
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size and self._access_order:
            # Evict LRU
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
            
        self._cache[key] = (result, time.time())
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        self._evict_expired()
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "hit_count": self.hits,
            "miss_count": self.misses,
            "hit_rate": hit_rate,
        }
