"""
Evaluation endpoints.
"""

import time
import os
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from rag.serving.schemas import EvaluationRequest, EvaluationResponse
from rag.serving.dependencies import get_settings, get_generation_engine
from rag.evaluation.test_set import TestSet
from rag.evaluation.evaluator import EvaluationRunner
from rag.evaluation.report import ReportGenerator


router = APIRouter()


@router.post("/evaluation/run", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest):
    """
    Run the evaluation pipeline on a test set.
    Returns aggregate metrics and saves a full report.
    """
    path = request.test_set_path
    if path is None:
        settings = get_settings()
        # Default test set path if not configured
        path = getattr(settings.evaluation, 'test_set_path', "./rag/data/eval/test_set.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Test set not found at {path}")

    test_set = TestSet.load(path)

    t0 = time.time()
    
    engine = get_generation_engine()
    settings = get_settings()
    
    runner = EvaluationRunner(
        settings=settings,
        generation_engine=engine,
    )

    report = runner.evaluate_test_set(
        test_set,
        k_values=request.k_values,
        progress=False,
    )
    
    elapsed = time.time() - t0

    report_gen = ReportGenerator(settings=settings)
    report_text = report_gen.generate_report(report)
    report_path = report_gen.save_report(report_text)

    return EvaluationResponse(
        status="completed",
        total_questions=report.total_questions,
        aggregate_metrics=report.aggregate_metrics,
        report_path=report_path,
        processing_time_seconds=round(elapsed, 2),
    )


@router.get("/evaluation/report")
async def get_report():
    """
    Get the most recent evaluation report as markdown.
    """
    settings = get_settings()
    # Default report path if not configured
    report_path = getattr(settings.evaluation, 'report_path', "./rag/data/eval/report.md")

    if not os.path.exists(report_path):
        raise FileNotFoundError("No evaluation report found. Run /evaluation/run first.")

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    return PlainTextResponse(content, media_type="text/markdown")
