import pytest
from rag.evaluation.metrics.generation_metrics import GenerationMetrics


class TestGenerationMetrics:
    def test_keyword_coverage_perfect(self):
        metrics = GenerationMetrics()
        answer = "This is a Test answer."
        assert metrics.keyword_coverage(answer, ["test", "answer"]) == 1.0

    def test_keyword_coverage_partial(self):
        metrics = GenerationMetrics()
        answer = "This is a test."
        assert metrics.keyword_coverage(answer, ["test", "answer"]) == 0.5

    def test_keyword_coverage_empty_keywords(self):
        metrics = GenerationMetrics()
        answer = "This is a test."
        assert metrics.keyword_coverage(answer, []) == 1.0

    def test_keyword_coverage_case_insensitive(self):
        metrics = GenerationMetrics()
        answer = "hello WORLD"
        assert metrics.keyword_coverage(answer, ["HELLO", "world"]) == 1.0

    def test_faithfulness_high(self):
        metrics = GenerationMetrics()
        answer = "The server runs on port 8080. It uses JSON."
        context = "Default configuration: server port 8080. Data format is JSON."
        assert metrics.faithfulness(answer, context) == 1.0

    def test_faithfulness_low(self):
        metrics = GenerationMetrics()
        answer = "The application executes on endpoint 9090. It utilizes XML."
        context = "Default configuration: server port 8080. Data format is JSON."
        assert metrics.faithfulness(answer, context) == 0.0

    def test_faithfulness_empty_answer(self):
        metrics = GenerationMetrics()
        assert metrics.faithfulness("", "context") == 1.0

    def test_answer_relevance_high(self):
        metrics = GenerationMetrics()
        answer = "The port is 8080."
        question = "What is the port?"
        assert metrics.answer_relevance(answer, question) == 1.0

    def test_answer_relevance_low(self):
        metrics = GenerationMetrics()
        answer = "I like apples."
        question = "What is the port?"
        assert metrics.answer_relevance(answer, question) == 0.0

    def test_answer_relevance_empty_question(self):
        metrics = GenerationMetrics()
        answer = "Yes."
        question = "Is it?" # 'is' and 'it' are stop words
        assert metrics.answer_relevance(answer, question) == 1.0

    def test_evaluate_generation_returns_all_fields(self):
        metrics = GenerationMetrics()
        results = metrics.evaluate_generation(
            answer="The port is 8080 [Source: Docs].",
            question="What is the port?",
            expected_keywords=["8080"],
            context="The port is 8080."
        )
        
        assert "keyword_coverage" in results
        assert "faithfulness" in results
        assert "answer_relevance" in results
        assert "answer_length_chars" in results
        assert "answer_length_words" in results
        assert "has_citations" in results
        assert "refused_answer" in results
        
        assert results["has_citations"] is True
        assert results["refused_answer"] is False

    def test_refused_answer_detected(self):
        metrics = GenerationMetrics()
        results = metrics.evaluate_generation(
            answer="I don't have enough information to answer.",
            question="Q?",
            expected_keywords=[]
        )
        
        assert results["refused_answer"] is True

    def test_has_citations_detected(self):
        metrics = GenerationMetrics()
        results = metrics.evaluate_generation(
            answer="Answer [Source: here].",
            question="Q?",
            expected_keywords=[]
        )
        
        assert results["has_citations"] is True
