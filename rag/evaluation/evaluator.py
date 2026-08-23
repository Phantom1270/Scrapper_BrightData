"""
Evaluation orchestrator.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json
import logging
from pathlib import Path

from rag.evaluation.test_set import EvalQuestion, TestSet
from rag.evaluation.metrics.retrieval_metrics import RetrievalMetrics
from rag.evaluation.metrics.generation_metrics import GenerationMetrics


logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    question: str
    category: str
    difficulty: str
    retrieved_chunk_ids: List[str]
    expected_chunk_ids: List[str]
    retrieval_metrics: dict
    generated_answer: str
    generation_metrics: dict
    confidence: str
    retrieval_time_ms: float
    generation_time_ms: float
    llm_model: str


@dataclass
class EvaluationReport:
    metadata: dict
    total_questions: int
    results: List[EvalResult]
    aggregate_metrics: dict
    category_metrics: dict
    difficulty_metrics: dict


class EvaluationRunner:
    """Orchestrates evaluation of the RAG pipeline."""

    def __init__(self, settings=None, generation_engine=None, retrieval_engine=None):
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()
            
        if generation_engine is None:
            from rag.generation.generator import GenerationEngine
            self.generation_engine = GenerationEngine(settings)
        else:
            self.generation_engine = generation_engine
            
        if retrieval_engine is None:
            self.retrieval_engine = self.generation_engine.retrieval_engine
        else:
            self.retrieval_engine = retrieval_engine
            
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        
        self.results_path = settings.evaluation.results_path
        self.report_path = settings.evaluation.report_path

    def evaluate_question(self, eval_question: EvalQuestion, k_values: List[int] = None) -> EvalResult:
        """Evaluate a single question through the full pipeline."""
        
        # 1. Run retrieval separately to get exact chunks
        results = self.retrieval_engine.retrieve(query=eval_question.question)
        retrieved_ids = [r.chunk_id for r in results]
        
        # 2. Compute retrieval metrics
        retrieval_scores = self.retrieval_metrics.evaluate_retrieval(
            retrieved_ids, eval_question.expected_chunk_ids, k_values
        )
        
        # 3. Run full generation
        gen_result = self.generation_engine.generate(query=eval_question.question)
        
        # 4. Build context for faithfulness check
        context = ""
        if results:
            from rag.generation.context_builder import ContextBuilder
            cb = ContextBuilder()
            context = cb.build_context(results)
            
        # 5. Compute generation metrics
        generation_scores = self.generation_metrics.evaluate_generation(
            answer=gen_result.answer,
            question=eval_question.question,
            expected_keywords=eval_question.expected_answer_keywords,
            context=context,
        )
        
        return EvalResult(
            question=eval_question.question,
            category=eval_question.category,
            difficulty=eval_question.difficulty,
            retrieved_chunk_ids=retrieved_ids,
            expected_chunk_ids=eval_question.expected_chunk_ids,
            retrieval_metrics=retrieval_scores,
            generated_answer=gen_result.answer,
            generation_metrics=generation_scores,
            confidence=gen_result.confidence,
            retrieval_time_ms=gen_result.retrieval_time_ms,
            generation_time_ms=gen_result.generation_time_ms,
            llm_model=gen_result.llm_model
        )

    def evaluate_test_set(self, test_set: TestSet, k_values: List[int] = None, progress: bool = True) -> EvaluationReport:
        """Evaluate all questions in a test set."""
        results = []
        total = len(test_set)
        
        for i, question in enumerate(test_set.questions):
            if progress:
                print(f"Evaluating {i+1}/{total}: {question.question}")
                
            result = self.evaluate_question(question, k_values)
            results.append(result)
            
        # Compute aggregate metrics
        aggregates = self._compute_aggregates(results)
        
        report = EvaluationReport(
            metadata=test_set.metadata,
            total_questions=total,
            results=results,
            aggregate_metrics=aggregates["total"],
            category_metrics=aggregates["category"],
            difficulty_metrics=aggregates["difficulty"]
        )
        
        # Save JSON results
        if self.results_path:
            self._save_results(report, self.results_path)
            
        return report

    def _compute_aggregates(self, results: List[EvalResult]) -> dict:
        """Compute average metrics across all results."""
        
        def average_group(group: List[EvalResult]) -> dict:
            if not group:
                return {}
                
            n = len(group)
            agg = {"count": n}
            
            # Aggregate retrieval metrics
            r_keys = group[0].retrieval_metrics.keys()
            for key in r_keys:
                if isinstance(group[0].retrieval_metrics[key], (int, float)):
                    agg[f"retrieval_{key}"] = sum(r.retrieval_metrics[key] for r in group) / n
                    
            # Aggregate generation metrics
            g_keys = group[0].generation_metrics.keys()
            for key in g_keys:
                val = group[0].generation_metrics[key]
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    # Handle optional metrics like faithfulness
                    valid_vals = [r.generation_metrics[key] for r in group if r.generation_metrics[key] is not None]
                    if valid_vals:
                        agg[f"generation_{key}"] = sum(valid_vals) / len(valid_vals)
                elif isinstance(val, bool):
                    agg[f"generation_{key}_rate"] = sum(1 for r in group if r.generation_metrics[key]) / n
                    
            return agg
            
        aggregates = {
            "total": average_group(results),
            "category": {},
            "difficulty": {}
        }
        
        # By category
        categories = set(r.category for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            aggregates["category"][cat] = average_group(cat_results)
            
        # By difficulty
        difficulties = set(r.difficulty for r in results)
        for diff in difficulties:
            diff_results = [r for r in results if r.difficulty == diff]
            aggregates["difficulty"][diff] = average_group(diff_results)
            
        return aggregates

    def _save_results(self, report: EvaluationReport, path: str) -> None:
        """Save raw results to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        data = asdict(report)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
