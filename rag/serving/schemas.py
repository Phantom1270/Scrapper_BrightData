"""
API schemas for request and response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryApiRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="The user's question")
    top_k: int = Field(default=5, ge=1, le=20,
                        description="Number of results to return")
    filter_content_type: Optional[str] = Field(
        default=None,
        description="Filter by content type: 'tutorial', 'api_reference', 'notebook'"
    )
    filter_doc_id: Optional[str] = Field(
        default=None,
        description="Filter by specific document ID"
    )
    use_reranking: bool = Field(
        default=True,
        description="Whether to apply cross-encoder re-ranking"
    )


class QueryApiResponse(BaseModel):
    answer: str
    sources: List[dict]
    citations: List[dict]
    confidence: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    cached: bool = False
    llm_model: str
    transform_used: Optional[str] = None
    reranker_used: Optional[str] = None


class IndexRequest(BaseModel):
    force_rebuild: bool = Field(
        default=False,
        description="Force full rebuild even if indexes exist"
    )


class IndexStatusResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    vector_store_count: int
    bm25_count: int
    last_indexed: Optional[str] = None


class EvaluationRequest(BaseModel):
    test_set_path: Optional[str] = Field(
        default=None,
        description="Path to test set JSON. Uses default if not provided."
    )
    k_values: Optional[List[int]] = Field(
        default=None,
        description="k values for metrics. Default: [1, 3, 5, 10]"
    )


class EvaluationResponse(BaseModel):
    status: str
    total_questions: int
    aggregate_metrics: dict
    report_path: str
    processing_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    components: dict


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int
