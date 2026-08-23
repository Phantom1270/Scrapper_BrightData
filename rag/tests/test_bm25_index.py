"""
Tests for BM25Index.
"""

import pytest
import os
from rag.search.bm25.bm25_index import BM25Index


class TestBM25Index:
    def test_build_and_search(self):
        index = BM25Index()
        index.build(
            ["1", "2", "3", "4", "5"],
            ["This is a test chunk.", "Another document here.", "Machine learning model.", "Pad 1", "Pad 2"]
        )
        results = index.search("test")
        assert len(results) > 0
        assert results[0][0] == "1"

    def test_search_returns_ranked_results(self):
        index = BM25Index()
        index.build(
            ["1", "2", "3", "4", "5", "6"],
            ["apple apple apple", "apple orange", "orange banana", "pad1", "pad2", "pad3"]
        )
        results = index.search("apple")
        assert len(results) == 2
        # "1" should rank higher than "2" because "apple" appears more times
        assert results[0][0] == "1"
        assert results[1][0] == "2"

    def test_search_exact_term_match(self):
        index = BM25Index()
        index.build(
            ["1", "2", "3", "4"],
            ["This uses sklearn.config_context for configuration.", "Just a generic config context.", "pad1", "pad2"]
        )
        results = index.search("config_context")
        assert len(results) > 0
        # Should match both potentially, but '1' has the exact 'config_context' token natively
        assert "1" in [r[0] for r in results]
        
    def test_tokenizer_handles_code_terms(self):
        index = BM25Index()
        tokens = index._tokenize("fit_transform and sklearn.config_context.")
        assert "fit_transform" in tokens
        assert "sklearn" in tokens
        assert "config_context" in tokens

    def test_save_and_load(self, tmp_path):
        index = BM25Index()
        index.build(["1", "2", "3", "4"], ["test document", "pad", "pad2", "pad3"])
        path = str(tmp_path / "bm25.pkl")
        index.save(path)
        
        index2 = BM25Index()
        loaded = index2.load(path)
        assert loaded is True
        assert index2.count() == 4
        results = index2.search("test")
        assert len(results) == 1
        assert results[0][0] == "1"

    def test_load_nonexistent_file(self, tmp_path):
        index = BM25Index()
        loaded = index.load(str(tmp_path / "does_not_exist.pkl"))
        assert loaded is False

    def test_add_chunks_increases_count(self):
        index = BM25Index()
        index.build(["1", "2", "3", "4"], ["pad1", "pad2", "pad3", "pad4"])
        assert index.count() == 4
        index.add_chunks(["5"], ["test term"])
        assert index.count() == 5
        results = index.search("test")
        assert len(results) == 1
        assert results[0][0] == "5"

    def test_clear_resets_state(self):
        index = BM25Index()
        index.build(["1"], ["test"])
        assert index.count() == 1
        index.clear()
        assert index.count() == 0

    def test_empty_index_search(self):
        index = BM25Index()
        results = index.search("test")
        assert len(results) == 0

    def test_search_top_k_limit(self):
        index = BM25Index()
        
        texts = ["apple document" if i < 10 else "pad document" for i in range(100)]
        index.build(
            [str(i) for i in range(100)],
            texts
        )
        results = index.search("apple", top_k=5)
        assert len(results) == 5
