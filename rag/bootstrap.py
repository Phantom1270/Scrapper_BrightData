"""
Unified bootstrap script for the RAG pipeline.
"""

def create_pipeline(settings=None):
    """Create the full RAG pipeline from settings."""
    if settings is None:
        from rag.config.settings import get_settings
        settings = get_settings()

    # Storage
    import os
    db_path = os.path.join(settings.general.data_dir, "rag.db")
    from rag.storage.sqlite_store import SQLiteStore
    store = SQLiteStore(db_path)

    # Chunking
    from rag.chunking.engine import ChunkingEngine
    chunker = ChunkingEngine(settings, store)

    # Search Engine
    from rag.search.search_engine import SearchEngine
    search_engine = SearchEngine(settings, store=store)

    # Search / Indexing
    from rag.search.indexer import IndexBuilder
    indexer = IndexBuilder(settings, store=store, search_engine=search_engine)

    # Retrieval
    from rag.retrieval.retrieval_engine import RetrievalEngine
    retriever = RetrievalEngine(settings, search_engine=search_engine)

    # Generation
    from rag.generation.generator import GenerationEngine
    generator = GenerationEngine(settings, retrieval_engine=retriever)

    return {
        "settings": settings,
        "store": store,
        "chunker": chunker,
        "indexer": indexer,
        "retriever": retriever,
        "generator": generator,
    }
