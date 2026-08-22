"""
Tests for ChromaVectorStore.
"""

import pytest
from unittest.mock import Mock
import os
import shutil

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

if HAS_CHROMADB:
    from rag.search.vector_store.chroma_store import ChromaVectorStore


@pytest.fixture
def temp_chroma_settings(tmp_path):
    settings = Mock()
    settings.vector_store.persist_dir = str(tmp_path / "chroma")
    settings.vector_store.collection_name = "test_docs"
    return settings


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
class TestChromaVectorStore:
    def test_add_and_count_chroma(self, temp_chroma_settings):
        store = ChromaVectorStore(settings=temp_chroma_settings)
        store.add_chunks(["1", "2"], [[0.1, 0.2], [0.3, 0.4]], ["a", "b"], [{"k": "v1"}, {"k": "v2"}])
        assert store.count() == 2

    def test_upsert_behavior_chroma(self, temp_chroma_settings):
        store = ChromaVectorStore(settings=temp_chroma_settings)
        store.add_chunks(["1"], [[0.1, 0.2]], ["a"], [{"v": 1}])
        assert store.count() == 1
        
        # Upsert
        store.add_chunks(["1"], [[0.2, 0.3]], ["b"], [{"v": 2}])
        assert store.count() == 1
        
        chunk = store.get_chunk_by_id("1")
        assert chunk["document"] == "b"
        assert chunk["metadata"] == {"v": 2}

    def test_search_returns_results_chroma(self, temp_chroma_settings):
        store = ChromaVectorStore(settings=temp_chroma_settings)
        # Using simple vectors where dot product / cosine sim is clear
        store.add_chunks(["1", "2"], [[1.0, 0.0], [0.0, 1.0]], ["doc1", "doc2"], [{"k": "v1"}, {"k": "v2"}])
        
        results = store.search([1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[0]["score"] > 0.9

    def test_search_with_filter_chroma(self, temp_chroma_settings):
        store = ChromaVectorStore(settings=temp_chroma_settings)
        store.add_chunks(
            ["1", "2", "3"], 
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], 
            ["a", "b", "c"], 
            [{"content_type": "prose"}, {"content_type": "code"}, {"content_type": "prose"}]
        )
        results = store.search([1.0, 0.0], top_k=10, content_type_filter="prose")
        assert len(results) == 2
        for r in results:
            assert r["metadata"]["content_type"] == "prose"

    def test_delete_chunks_chroma(self, temp_chroma_settings):
        store = ChromaVectorStore(settings=temp_chroma_settings)
        store.add_chunks(["1", "2"], [[0.1, 0.2], [0.3, 0.4]], ["a", "b"], [{"k": "v1"}, {"k": "v2"}])
        store.delete_chunks(["1"])
        assert store.count() == 1
        assert store.get_chunk_by_id("2") is not None

    def test_delete_all_chroma(self, temp_chroma_settings):
        store = ChromaVectorStore(settings=temp_chroma_settings)
        store.add_chunks(["1", "2"], [[0.1, 0.2], [0.3, 0.4]], ["a", "b"], [{"k": "v1"}, {"k": "v2"}])
        store.delete_all()
        assert store.count() == 0

    def test_get_chunk_by_id_chroma(self, temp_chroma_settings):
        store = ChromaVectorStore(settings=temp_chroma_settings)
        store.add_chunks(["1"], [[0.1, 0.2]], ["a"], [{"k": "v"}])
        chunk = store.get_chunk_by_id("1")
        assert chunk is not None
        assert chunk["document"] == "a"

    def test_persistence_across_restarts(self, temp_chroma_settings):
        store1 = ChromaVectorStore(settings=temp_chroma_settings)
        store1.add_chunks(["1", "2"], [[0.1, 0.2], [0.3, 0.4]], ["a", "b"], [{"k": "v1"}, {"k": "v2"}])
        assert store1.count() == 2
        
        # Simulate restart by creating new instance pointing to same directory
        store2 = ChromaVectorStore(settings=temp_chroma_settings)
        assert store2.count() == 2
