"""
Dependency injection for FastAPI routes.
"""

# Module-level singletons (initialized on app startup)
_generation_engine = None
_retrieval_engine = None
_index_builder = None
_search_engine = None
_store = None
_cache = None
_settings = None
_start_time = None


def initialize_dependencies(settings=None) -> None:
    """Called during app startup. Creates all shared instances."""
    global _generation_engine, _retrieval_engine, _index_builder, _search_engine
    global _store, _cache, _settings, _start_time

    import time
    _start_time = time.time()

    if settings is None:
        from rag.config.settings import get_settings
        settings = get_settings()
    _settings = settings

    from rag.storage.sqlite_store import SQLiteStore
    import os
    db_path = os.path.join(settings.general.data_dir, "rag.db")
    _store = SQLiteStore(db_path)

    from rag.serving.cache import QueryCache
    _cache = QueryCache(
        ttl_seconds=settings.serving.cache_ttl_seconds,
        max_size=1000,
    )

    # Lazy initialization: engines are created on first use
    _generation_engine = None
    _retrieval_engine = None
    _index_builder = None
    _search_engine = None


def get_search_engine():
    """Get or create the SearchEngine singleton."""
    global _search_engine
    if _search_engine is None:
        from rag.search.search_engine import SearchEngine
        _search_engine = SearchEngine(settings=_settings, store=_store)
    return _search_engine


def get_generation_engine():
    """Get or create the GenerationEngine singleton."""
    global _generation_engine
    if _generation_engine is None:
        from rag.generation.generator import GenerationEngine
        _generation_engine = GenerationEngine(settings=_settings)
    return _generation_engine


def get_retrieval_engine():
    """Get or create the RetrievalEngine singleton."""
    global _retrieval_engine
    if _retrieval_engine is None:
        from rag.retrieval.retrieval_engine import RetrievalEngine
        _retrieval_engine = RetrievalEngine(settings=_settings, search_engine=get_search_engine())
    return _retrieval_engine


def get_index_builder():
    """Get or create the IndexBuilder singleton."""
    global _index_builder
    if _index_builder is None:
        from rag.search.indexer import IndexBuilder
        _index_builder = IndexBuilder(settings=_settings, store=_store, search_engine=get_search_engine())
    return _index_builder


def get_store():
    return _store


def get_cache():
    return _cache


def get_settings():
    return _settings


def get_start_time():
    return _start_time


def shutdown_dependencies() -> None:
    """Called during app shutdown. Cleanup resources."""
    global _generation_engine, _retrieval_engine, _index_builder, _search_engine
    global _store, _cache
    _generation_engine = None
    _retrieval_engine = None
    _index_builder = None
    _search_engine = None
    _cache = None
    # Store can persist — it's just a SQLite file.
