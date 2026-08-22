"""
Tests for base vector store using a mock implementation.
"""

from typing import List, Optional, Dict
import pytest
import math

from rag.search.vector_store.base import BaseVectorStore


class MockVectorStore(BaseVectorStore):
    def __init__(self):
        self.store = {}

    def add_chunks(self, chunk_ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict]) -> None:
        for cid, emb, doc, meta in zip(chunk_ids, embeddings, documents, metadatas):
            self.store[cid] = {
                "id": cid,
                "embedding": emb,
                "document": doc,
                "metadata": meta
            }

    def _cosine_sim(self, v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(self, query_embedding: List[float], top_k: int = 10, content_type_filter: str = None, doc_id_filter: str = None) -> List[Dict]:
        results = []
        for cid, data in self.store.items():
            if content_type_filter and data["metadata"].get("content_type") != content_type_filter:
                continue
            if doc_id_filter and data["metadata"].get("doc_id") != doc_id_filter:
                continue
            score = self._cosine_sim(query_embedding, data["embedding"])
            results.append({
                "id": cid,
                "score": score,
                "document": data["document"],
                "metadata": data["metadata"]
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        for cid in chunk_ids:
            if cid in self.store:
                del self.store[cid]

    def delete_all(self) -> None:
        self.store = {}

    def count(self) -> int:
        return len(self.store)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        return self.store.get(chunk_id)


class TestBaseVectorStore:
    def test_add_and_count(self):
        store = MockVectorStore()
        store.add_chunks(["1", "2", "3"], [[0.1], [0.2], [0.3]], ["a", "b", "c"], [{}, {}, {}])
        assert store.count() == 3

    def test_add_upsert_behavior(self):
        store = MockVectorStore()
        store.add_chunks(["1"], [[0.1]], ["a"], [{"v": 1}])
        assert store.count() == 1
        
        # Upsert
        store.add_chunks(["1"], [[0.2]], ["b"], [{"v": 2}])
        assert store.count() == 1
        
        chunk = store.get_chunk_by_id("1")
        assert chunk["document"] == "b"
        assert chunk["embedding"] == [0.2]

    def test_search_returns_top_k(self):
        store = MockVectorStore()
        ids = [str(i) for i in range(10)]
        embs = [[float(i)] for i in range(10)]
        docs = [str(i) for i in range(10)]
        metas = [{} for _ in range(10)]
        store.add_chunks(ids, embs, docs, metas)
        
        results = store.search([5.0], top_k=3)
        assert len(results) == 3

    def test_search_returns_sorted_by_score(self):
        store = MockVectorStore()
        store.add_chunks(["1", "2"], [[1.0, 0.0], [0.0, 1.0]], ["doc1", "doc2"], [{}, {}])
        
        # query closer to doc1
        results = store.search([0.9, 0.1], top_k=2)
        assert results[0]["id"] == "1"
        assert results[1]["id"] == "2"
        assert results[0]["score"] > results[1]["score"]

    def test_search_with_content_type_filter(self):
        store = MockVectorStore()
        store.add_chunks(
            ["1", "2", "3"], 
            [[1.0], [1.0], [1.0]], 
            ["a", "b", "c"], 
            [{"content_type": "prose"}, {"content_type": "code"}, {"content_type": "prose"}]
        )
        results = store.search([1.0], top_k=10, content_type_filter="prose")
        assert len(results) == 2
        for r in results:
            assert r["metadata"]["content_type"] == "prose"

    def test_search_with_doc_id_filter(self):
        store = MockVectorStore()
        store.add_chunks(
            ["1", "2", "3"], 
            [[1.0], [1.0], [1.0]], 
            ["a", "b", "c"], 
            [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d1"}]
        )
        results = store.search([1.0], top_k=10, doc_id_filter="d1")
        assert len(results) == 2
        for r in results:
            assert r["metadata"]["doc_id"] == "d1"

    def test_delete_chunks(self):
        store = MockVectorStore()
        store.add_chunks(["1", "2", "3"], [[0.1], [0.2], [0.3]], ["a", "b", "c"], [{}, {}, {}])
        store.delete_chunks(["1", "3"])
        assert store.count() == 1
        assert store.get_chunk_by_id("2") is not None

    def test_delete_all(self):
        store = MockVectorStore()
        store.add_chunks(["1", "2"], [[0.1], [0.2]], ["a", "b"], [{}, {}])
        store.delete_all()
        assert store.count() == 0

    def test_get_chunk_by_id_found(self):
        store = MockVectorStore()
        store.add_chunks(["1"], [[0.1]], ["a"], [{"k": "v"}])
        chunk = store.get_chunk_by_id("1")
        assert chunk is not None
        assert chunk["document"] == "a"
        assert chunk["metadata"] == {"k": "v"}

    def test_get_chunk_by_id_not_found(self):
        store = MockVectorStore()
        assert store.get_chunk_by_id("missing") is None

    def test_search_empty_store(self):
        store = MockVectorStore()
        results = store.search([1.0], top_k=10)
        assert len(results) == 0
