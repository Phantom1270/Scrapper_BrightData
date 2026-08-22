import pytest
from unittest.mock import patch, MagicMock

from rag.serving.dependencies import (
    initialize_dependencies, shutdown_dependencies,
    get_store, get_cache, get_generation_engine, get_retrieval_engine
)
from rag.storage.sqlite_store import SQLiteStore
from rag.serving.cache import QueryCache


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.serving.cache_ttl_seconds = 300
    settings.general.data_dir = "./test_data"
    return settings


class TestDependencies:
    def test_initialize_dependencies(self, mock_settings):
        # We need to ensure we don't actually create databases in tests
        with patch('rag.storage.sqlite_store.SQLiteStore.__init__', return_value=None):
            initialize_dependencies(mock_settings)
            
            store = get_store()
            assert isinstance(store, SQLiteStore)
            
            cache = get_cache()
            assert isinstance(cache, QueryCache)
            
            shutdown_dependencies()

    def test_get_cache_returns_cache(self, mock_settings):
        with patch('rag.storage.sqlite_store.SQLiteStore.__init__', return_value=None):
            initialize_dependencies(mock_settings)
            assert isinstance(get_cache(), QueryCache)
            shutdown_dependencies()

    def test_shutdown_dependencies(self, mock_settings):
        with patch('rag.storage.sqlite_store.SQLiteStore.__init__', return_value=None):
            initialize_dependencies(mock_settings)
            shutdown_dependencies()
            
            # Since shutdown nullifies them:
            assert get_store() is not None # Store remains
            assert get_cache() is None

    @patch('rag.generation.generator.GenerationEngine')
    def test_get_generation_engine_lazy_init(self, MockEngine, mock_settings):
        with patch('rag.storage.sqlite_store.SQLiteStore.__init__', return_value=None):
            initialize_dependencies(mock_settings)
            
            engine1 = get_generation_engine()
            engine2 = get_generation_engine()
            
            assert engine1 is engine2
            MockEngine.assert_called_once()
            
            shutdown_dependencies()

    @patch('rag.retrieval.retrieval_engine.RetrievalEngine')
    def test_get_retrieval_engine_lazy_init(self, MockEngine, mock_settings):
        with patch('rag.storage.sqlite_store.SQLiteStore.__init__', return_value=None):
            initialize_dependencies(mock_settings)
            
            engine1 = get_retrieval_engine()
            engine2 = get_retrieval_engine()
            
            assert engine1 is engine2
            MockEngine.assert_called_once()
            
            shutdown_dependencies()
