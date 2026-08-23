import pytest
from unittest.mock import MagicMock, patch

from rag.evaluation.evaluator import EvaluationRunner, EvalResult, EvaluationReport
from rag.evaluation.test_set import EvalQuestion, TestSet
from rag.models.retrieval import RetrievalResult
from rag.generation.generator import GenerationResult


@pytest.fixture
def sample_test_set():
    questions = [
        EvalQuestion("Q1", ["k1"], ["c1"], ["api"], "api", "easy"),
        EvalQuestion("Q2", ["k2"], ["c2"], ["tut"], "tutorial", "medium"),
        EvalQuestion("Q3", ["k3"], ["c3"], ["api"], "api", "hard"),
    ]
    return TestSet(questions, metadata={"name": "test"})


@pytest.fixture
def mock_retrieval_engine():
    engine = MagicMock()
    # Mock retrieve to return something for context builder
    results = [
        RetrievalResult("c1", "content 1", "url1", ["H1"], "api", 0.9, "vector"),
        RetrievalResult("c2", "content 2", "url2", ["H2"], "tut", 0.8, "vector")
    ]
    engine.retrieve.return_value = results
    return engine


@pytest.fixture
def mock_generation_engine(mock_retrieval_engine):
    engine = MagicMock()
    # It must have the retrieval engine attached
    engine.retrieval_engine = mock_retrieval_engine
    
    result = GenerationResult(
        answer="This is a mock answer with k1.",
        sources=[],
        citations=[],
        confidence="high",
        retrieval_time_ms=10.0,
        generation_time_ms=50.0,
        chunks_retrieved=2,
        chunks_used_in_context=2,
        llm_model="mock"
    )
    engine.generate.return_value = result
    return engine


@pytest.fixture
def mock_settings(tmp_path):
    settings = MagicMock()
    settings.evaluation.results_path = str(tmp_path / "results.json")
    settings.evaluation.report_path = str(tmp_path / "report.md")
    settings.generation.max_context_tokens = 3000
    settings.chunking.encoding_name = "cl100k_base"
    return settings


class TestEvaluationRunner:
    def test_evaluate_question_returns_eval_result(self, mock_settings, mock_generation_engine):
        runner = EvaluationRunner(settings=mock_settings, generation_engine=mock_generation_engine)
        question = EvalQuestion("Q1", ["k1"], ["c1"], ["api"], "api", "easy")
        
        result = runner.evaluate_question(question)
        
        assert isinstance(result, EvalResult)
        assert result.question == "Q1"
        assert result.category == "api"
        assert result.difficulty == "easy"
        assert result.generated_answer == "This is a mock answer with k1."
        assert result.confidence == "high"

    def test_evaluate_question_retrieval_metrics_populated(self, mock_settings, mock_generation_engine):
        runner = EvaluationRunner(settings=mock_settings, generation_engine=mock_generation_engine)
        question = EvalQuestion("Q1", ["k1"], ["c1"], ["api"], "api", "easy")
        
        result = runner.evaluate_question(question)
        
        assert "retrieved_count" in result.retrieval_metrics
        assert "overlap_count" in result.retrieval_metrics
        assert "mrr" in result.retrieval_metrics
        assert "precision@1" in result.retrieval_metrics
        assert result.retrieval_metrics["retrieved_count"] == 2 # 2 mock results

    def test_evaluate_question_generation_metrics_populated(self, mock_settings, mock_generation_engine):
        runner = EvaluationRunner(settings=mock_settings, generation_engine=mock_generation_engine)
        question = EvalQuestion("Q1", ["k1"], ["c1"], ["api"], "api", "easy")
        
        result = runner.evaluate_question(question)
        
        assert "keyword_coverage" in result.generation_metrics
        assert "faithfulness" in result.generation_metrics
        assert "answer_relevance" in result.generation_metrics
        assert "has_citations" in result.generation_metrics

    def test_evaluate_test_set_returns_report(self, mock_settings, mock_generation_engine, sample_test_set):
        runner = EvaluationRunner(settings=mock_settings, generation_engine=mock_generation_engine)
        
        report = runner.evaluate_test_set(sample_test_set, progress=False)
        
        assert isinstance(report, EvaluationReport)
        assert report.total_questions == 3
        assert len(report.results) == 3

    def test_evaluate_test_set_aggregate_metrics(self, mock_settings, mock_generation_engine, sample_test_set):
        runner = EvaluationRunner(settings=mock_settings, generation_engine=mock_generation_engine)
        
        report = runner.evaluate_test_set(sample_test_set, progress=False)
        
        assert "count" in report.aggregate_metrics
        assert report.aggregate_metrics["count"] == 3
        assert "retrieval_mrr" in report.aggregate_metrics
        assert "generation_keyword_coverage" in report.aggregate_metrics

    def test_evaluate_test_set_category_metrics(self, mock_settings, mock_generation_engine, sample_test_set):
        runner = EvaluationRunner(settings=mock_settings, generation_engine=mock_generation_engine)
        
        report = runner.evaluate_test_set(sample_test_set, progress=False)
        
        assert "api" in report.category_metrics
        assert "tutorial" in report.category_metrics
        assert report.category_metrics["api"]["count"] == 2
        assert report.category_metrics["tutorial"]["count"] == 1

    def test_evaluate_test_set_saves_results(self, mock_settings, mock_generation_engine, sample_test_set):
        runner = EvaluationRunner(settings=mock_settings, generation_engine=mock_generation_engine)
        
        runner.evaluate_test_set(sample_test_set, progress=False)
        
        import os
        assert os.path.exists(mock_settings.evaluation.results_path)

    def test_evaluate_test_set_progress_output(self, mock_settings, mock_generation_engine, sample_test_set, capsys):
        runner = EvaluationRunner(settings=mock_settings, generation_engine=mock_generation_engine)
        
        runner.evaluate_test_set(sample_test_set, progress=True)
        
        captured = capsys.readouterr()
        assert "Evaluating 1/3: Q1" in captured.out
        assert "Evaluating 3/3: Q3" in captured.out
