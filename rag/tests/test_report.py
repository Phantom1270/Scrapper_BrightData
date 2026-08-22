import pytest
import os
from unittest.mock import MagicMock

from rag.evaluation.report import ReportGenerator
from rag.evaluation.evaluator import EvaluationReport, EvalResult


@pytest.fixture
def mock_settings(tmp_path):
    settings = MagicMock()
    settings.evaluation.report_path = str(tmp_path / "report.md")
    return settings


@pytest.fixture
def sample_report():
    results = [
        EvalResult(
            question="Q1", category="api", difficulty="easy",
            retrieved_chunk_ids=["c1", "c2"], expected_chunk_ids=["c1"],
            retrieval_metrics={"precision@5": 1.0, "recall@5": 1.0, "mrr": 1.0, "overlap_count": 1},
            generated_answer="Answer 1",
            generation_metrics={"keyword_coverage": 1.0, "faithfulness": 1.0, "refused_answer": False},
            confidence="high", retrieval_time_ms=10.0, generation_time_ms=20.0,
            llm_model="test-model"
        ),
        EvalResult(
            question="Q2", category="tutorial", difficulty="medium",
            retrieved_chunk_ids=[], expected_chunk_ids=["c3"],
            retrieval_metrics={"precision@5": 0.0, "recall@5": 0.0, "mrr": 0.0, "overlap_count": 0},
            generated_answer="I don't know.",
            generation_metrics={"keyword_coverage": 0.0, "faithfulness": 1.0, "refused_answer": True},
            confidence="none", retrieval_time_ms=5.0, generation_time_ms=10.0,
            llm_model="test-model"
        ),
        EvalResult(
            question="Q3", category="api", difficulty="hard",
            retrieved_chunk_ids=["c4"], expected_chunk_ids=["c4"],
            retrieval_metrics={"precision@5": 1.0, "recall@5": 1.0, "mrr": 1.0, "overlap_count": 1},
            generated_answer="Answer 3 " * 50, # long answer
            generation_metrics={"keyword_coverage": 0.8, "faithfulness": 0.9, "refused_answer": False},
            confidence="low", retrieval_time_ms=15.0, generation_time_ms=30.0,
            llm_model="test-model"
        )
    ]
    
    aggregate_metrics = {
        "retrieval_precision@5": 0.666,
        "retrieval_recall@5": 0.666,
        "retrieval_mrr": 0.666,
        "generation_keyword_coverage": 0.6,
        "generation_faithfulness": 0.966,
        "generation_has_citations_rate": 0.0,
        "generation_generation_time_ms": 20.0
    }
    
    category_metrics = {
        "api": {"count": 2, "retrieval_precision@5": 1.0, "generation_keyword_coverage": 0.9},
        "tutorial": {"count": 1, "retrieval_precision@5": 0.0, "generation_keyword_coverage": 0.0}
    }
    
    difficulty_metrics = {
        "easy": {"count": 1, "retrieval_precision@5": 1.0, "generation_keyword_coverage": 1.0},
        "medium": {"count": 1, "retrieval_precision@5": 0.0, "generation_keyword_coverage": 0.0},
        "hard": {"count": 1, "retrieval_precision@5": 1.0, "generation_keyword_coverage": 0.8},
    }
    
    return EvaluationReport(
        metadata={"name": "Test Report"},
        total_questions=3,
        results=results,
        aggregate_metrics=aggregate_metrics,
        category_metrics=category_metrics,
        difficulty_metrics=difficulty_metrics
    )


class TestReportGenerator:
    def test_generate_report_returns_string(self, sample_report, mock_settings):
        generator = ReportGenerator(settings=mock_settings)
        report_text = generator.generate_report(sample_report)
        
        assert isinstance(report_text, str)
        assert len(report_text) > 100

    def test_generate_report_contains_header(self, sample_report, mock_settings):
        generator = ReportGenerator(settings=mock_settings)
        report_text = generator.generate_report(sample_report)
        
        assert "# RAG Pipeline Evaluation Report" in report_text
        assert "Test Set:** Test Report" in report_text
        assert "Total Questions:** 3" in report_text
        assert "LLM Model:** test-model" in report_text

    def test_generate_report_contains_overall_metrics_table(self, sample_report, mock_settings):
        generator = ReportGenerator(settings=mock_settings)
        report_text = generator.generate_report(sample_report)
        
        assert "| Metric | Value |" in report_text
        assert "| Precision@5 | 0.666 |" in report_text
        assert "| Keyword Coverage | 0.600 |" in report_text

    def test_generate_report_contains_category_sections(self, sample_report, mock_settings):
        generator = ReportGenerator(settings=mock_settings)
        report_text = generator.generate_report(sample_report)
        
        assert "### Api Questions (2 questions)" in report_text
        assert "### Tutorial Questions (1 questions)" in report_text

    def test_generate_report_contains_individual_results(self, sample_report, mock_settings):
        generator = ReportGenerator(settings=mock_settings)
        report_text = generator.generate_report(sample_report)
        
        assert "### Q1: Q1" in report_text
        assert "### Q2: Q2" in report_text
        assert "### Q3: Q3" in report_text
        
        # Check truncation of long answer
        assert "Answer 3 Answer 3" in report_text
        assert "..." in report_text

    def test_generate_report_contains_observations(self, sample_report, mock_settings):
        generator = ReportGenerator(settings=mock_settings)
        report_text = generator.generate_report(sample_report)
        
        assert "## Observations" in report_text
        assert "Best performing category (Precision@5): **api**" in report_text
        assert "Worst performing category (Precision@5): **tutorial**" in report_text
        assert "Warning: Category **tutorial** has poor retrieval performance" in report_text
        assert "Questions with zero retrieval overlap: **1**" in report_text
        assert "Questions where the LLM refused to answer: **1**" in report_text

    def test_save_report_creates_file(self, tmp_path, sample_report, mock_settings):
        generator = ReportGenerator(settings=mock_settings)
        report_text = generator.generate_report(sample_report)
        
        file_path = tmp_path / "report.md"
        generator.save_report(report_text, str(file_path))
        
        assert os.path.exists(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "## Observations" in content

    def test_save_report_creates_parent_directories(self, tmp_path, sample_report, mock_settings):
        generator = ReportGenerator(settings=mock_settings)
        report_text = generator.generate_report(sample_report)
        
        file_path = tmp_path / "deep" / "nested" / "dir" / "report.md"
        generator.save_report(report_text, str(file_path))
        
        assert os.path.exists(file_path)

    def test_generate_report_handles_zero_metrics(self, mock_settings):
        empty_report = EvaluationReport(
            metadata={},
            total_questions=0,
            results=[],
            aggregate_metrics={},
            category_metrics={},
            difficulty_metrics={}
        )
        
        generator = ReportGenerator(settings=mock_settings)
        report_text = generator.generate_report(empty_report)
        
        # Should not throw divide by zero errors
        assert "N/A" in report_text or "0.000" in report_text
        assert "No results to observe." in report_text
