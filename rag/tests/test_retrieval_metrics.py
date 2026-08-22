import pytest
from rag.evaluation.metrics.retrieval_metrics import RetrievalMetrics


class TestRetrievalMetrics:
    def test_precision_at_k_perfect(self):
        metrics = RetrievalMetrics()
        assert metrics.precision_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0

    def test_precision_at_k_partial(self):
        metrics = RetrievalMetrics()
        assert metrics.precision_at_k(["a", "x", "y"], ["a", "b", "c"], 3) == pytest.approx(0.3333333333333333)

    def test_precision_at_k_zero(self):
        metrics = RetrievalMetrics()
        assert metrics.precision_at_k(["x", "y", "z"], ["a", "b", "c"], 3) == 0.0

    def test_recall_at_k_perfect(self):
        metrics = RetrievalMetrics()
        assert metrics.recall_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0

    def test_recall_at_k_partial(self):
        metrics = RetrievalMetrics()
        assert metrics.recall_at_k(["a", "x"], ["a", "b", "c"], 2) == pytest.approx(0.3333333333333333)

    def test_recall_at_k_empty_expected(self):
        metrics = RetrievalMetrics()
        assert metrics.recall_at_k(["a", "b"], [], 2) == 0.0

    def test_mrr_first_position(self):
        metrics = RetrievalMetrics()
        assert metrics.mean_reciprocal_rank(["a", "x", "y"], ["a"]) == 1.0

    def test_mrr_second_position(self):
        metrics = RetrievalMetrics()
        assert metrics.mean_reciprocal_rank(["x", "a", "y"], ["a"]) == 0.5

    def test_mrr_not_found(self):
        metrics = RetrievalMetrics()
        assert metrics.mean_reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0

    def test_ndcg_perfect(self):
        metrics = RetrievalMetrics()
        assert metrics.ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0

    def test_ndcg_imperfect(self):
        metrics = RetrievalMetrics()
        ndcg = metrics.ndcg_at_k(["a", "x", "b"], ["a", "b"], 3)
        assert 0.0 < ndcg < 1.0

    def test_ndcg_empty_expected(self):
        metrics = RetrievalMetrics()
        assert metrics.ndcg_at_k(["a", "b"], [], 3) == 0.0

    def test_evaluate_retrieval_returns_all_metrics(self):
        metrics = RetrievalMetrics()
        results = metrics.evaluate_retrieval(["a", "b", "c"], ["a", "c"])
        
        assert "retrieved_count" in results
        assert "expected_count" in results
        assert "overlap_count" in results
        assert "mrr" in results
        assert "precision@1" in results
        assert "precision@5" in results
        assert "recall@3" in results
        assert "ndcg@10" in results
        
        assert results["overlap_count"] == 2

    def test_evaluate_retrieval_custom_k_values(self):
        metrics = RetrievalMetrics()
        results = metrics.evaluate_retrieval(["a", "b"], ["a"], k_values=[1, 5, 20])
        
        assert "precision@1" in results
        assert "precision@5" in results
        assert "precision@20" in results
        assert "precision@3" not in results
