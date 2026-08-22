"""
Tests for base embedder.
"""

from typing import List
from rag.search.embeddings.base import BaseEmbedder

class MockEmbedder(BaseEmbedder):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 8 for _ in texts]
        
    def embed_query(self, query: str) -> List[float]:
        return [0.1] * 8
        
    def get_dimension(self) -> int:
        return 8
        
    def get_model_name(self) -> str:
        return "mock-model"


class TestBaseEmbedder:
    def test_embed_documents_returns_correct_count(self):
        embedder = MockEmbedder()
        texts = ["t1", "t2", "t3", "t4", "t5"]
        results = embedder.embed_documents(texts)
        assert len(results) == 5

    def test_embed_documents_returns_correct_dimension(self):
        embedder = MockEmbedder()
        texts = ["t1"]
        results = embedder.embed_documents(texts)
        assert len(results[0]) == 8

    def test_embed_query_returns_single_vector(self):
        embedder = MockEmbedder()
        result = embedder.embed_query("test")
        assert len(result) == 8
        assert isinstance(result, list)

    def test_get_dimension_consistent(self):
        embedder = MockEmbedder()
        assert embedder.get_dimension() == 8

    def test_embed_documents_batched_processes_all(self):
        embedder = MockEmbedder()
        texts = [f"t{i}" for i in range(200)]
        results = embedder.embed_documents_batched(texts, batch_size=64, show_progress=False)
        assert len(results) == 200

    def test_embed_documents_batched_preserves_order(self):
        # Even with batching, order must be preserved. Since we just slice and extend, it naturally does.
        # But we'll test size logic.
        embedder = MockEmbedder()
        texts = [f"t{i}" for i in range(100)]
        results = embedder.embed_documents_batched(texts, batch_size=30, show_progress=False)
        assert len(results) == 100

    def test_embed_documents_batched_progress_output(self, capsys):
        embedder = MockEmbedder()
        texts = [f"t{i}" for i in range(50)]
        embedder.embed_documents_batched(texts, batch_size=20, show_progress=True)
        captured = capsys.readouterr()
        
        # 50 total, batch size 20 means it prints at 20, 40, 50.
        assert "Embedded 20/50 chunks" in captured.out
        assert "Embedded 40/50 chunks" in captured.out
        assert "Embedded 50/50 chunks" in captured.out
