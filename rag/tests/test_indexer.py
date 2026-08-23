"""
Tests for IndexBuilder.
"""

import pytest
import os
import json
from unittest.mock import Mock, patch

from rag.search.indexer import IndexBuilder, IndexBuildResult
from rag.search.bm25.bm25_index import BM25Index
from rag.models.chunk import Chunk


class MockEmbedder:
    def embed_documents_batched(self, texts, batch_size=None, show_progress=False):
        return [[0.1] * 8 for _ in texts]
        
    def get_dimension(self):
        return 8
        
    def get_model_name(self):
        return "mock-model"


class MockVectorStore:
    def __init__(self):
        self.store = {}
        
    def add_chunks(self, ids, embs, docs, metas):
        for idx, e, d, m in zip(ids, embs, docs, metas):
            self.store[idx] = {"emb": e, "doc": d, "meta": m}
            
    def count(self):
        return len(self.store)
        
    def delete_all(self):
        self.store = {}


class MockStore:
    def __init__(self):
        self.chunks = {}
        
    def add_chunk(self, chunk):
        self.chunks[chunk.chunk_id] = chunk
        
    def get_all_chunks(self):
        return list(self.chunks.values())
        
    def get_chunk(self, chunk_id):
        return self.chunks.get(chunk_id)


@pytest.fixture
def setup_indexer(tmp_path):
    settings = Mock()
    settings.general.data_dir = str(tmp_path)
    settings.bm25.index_path = str(tmp_path / "bm25.pkl")
    
    store = MockStore()
    chunk1 = Chunk("c1", "d1", "url1", "c1 text", "prose", ["H1"], 0, 5)
    chunk2 = Chunk("c2", "d1", "url1", "c2 text", "code", ["H1"], 1, 10, metadata={"k": "v"})
    store.add_chunk(chunk1)
    store.add_chunk(chunk2)
    
    embedder = MockEmbedder()
    vector_store = MockVectorStore()
    bm25_index = BM25Index()
    
    builder = IndexBuilder(
        settings=settings,
        store=store,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index
    )
    return builder, store, vector_store, bm25_index, tmp_path


class TestIndexBuilder:
    def test_build_all_indexes_all_chunks(self, setup_indexer):
        builder, store, vector_store, bm25_index, tmp_path = setup_indexer
        result = builder.build_all(force_rebuild=True)
        
        assert result.total_chunks == 2
        assert result.chunks_embedded == 2
        assert vector_store.count() == 2
        assert bm25_index.count() == 2
        assert result.skipped is False
        
    def test_build_all_creates_manifest(self, setup_indexer):
        builder, _, _, _, tmp_path = setup_indexer
        result = builder.build_all(force_rebuild=True)
        
        manifest_path = tmp_path / "indexes" / "manifest.json"
        assert manifest_path.exists()
        
        with open(manifest_path, "r") as f:
            data = json.load(f)
            
        assert data["total_chunks"] == 2
        assert data["embedding_model"] == "mock-model"
        assert data["embedding_dimension"] == 8
        assert "built_at" in data

    def test_build_all_force_rebuild(self, setup_indexer):
        builder, _, _, _, _ = setup_indexer
        builder.build_all(force_rebuild=True)
        # Second time with force rebuild
        result = builder.build_all(force_rebuild=True)
        assert result.skipped is False
        assert result.chunks_embedded == 2

    def test_build_all_skips_if_existing(self, setup_indexer):
        builder, _, _, _, _ = setup_indexer
        builder.build_all(force_rebuild=True)
        # Second time without force rebuild
        result = builder.build_all(force_rebuild=False)
        assert result.skipped is True
        assert result.chunks_embedded == 0

    def test_build_incremental_adds_new_chunks(self, setup_indexer):
        builder, store, vector_store, bm25_index, _ = setup_indexer
        builder.build_all(force_rebuild=True)
        
        # Add new chunk to store
        chunk3 = Chunk("c3", "d2", "url2", "c3 text", "prose", ["H2"], 0, 5)
        store.add_chunk(chunk3)
        
        result = builder.build_incremental()
        assert vector_store.count() == 3
        assert bm25_index.count() == 3
        assert result.chunks_embedded == 1
        assert result.incremental is True

    def test_build_incremental_no_new_chunks(self, setup_indexer):
        builder, _, _, _, _ = setup_indexer
        builder.build_all(force_rebuild=True)
        
        result = builder.build_incremental()
        assert result.skipped is True
        assert result.chunks_embedded == 0

    def test_build_all_empty_store(self, tmp_path):
        settings = Mock()
        settings.general.data_dir = str(tmp_path)
        settings.bm25.index_path = str(tmp_path / "bm25.pkl")
        store = MockStore()
        builder = IndexBuilder(settings=settings, store=store, embedder=MockEmbedder(), vector_store=MockVectorStore(), bm25_index=BM25Index())
        
        result = builder.build_all(force_rebuild=True)
        assert result.total_chunks == 0
        assert result.chunks_embedded == 0

    def test_load_existing(self, setup_indexer):
        builder, _, _, _, _ = setup_indexer
        builder.build_all(force_rebuild=True)
        
        assert builder.load_existing() is True
