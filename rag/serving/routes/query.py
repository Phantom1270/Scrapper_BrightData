"""
Query endpoint.
"""

from fastapi import APIRouter
from rag.serving.schemas import QueryApiRequest, QueryApiResponse
from rag.serving.dependencies import get_cache, get_generation_engine
from rag.models.query import QueryRequest


router = APIRouter()


@router.post("/query", response_model=QueryApiResponse)
async def query(request: QueryApiRequest):
    """
    Ask a question about the indexed documentation.
    Returns a grounded answer with citations and source chunks.
    """
    cache = get_cache()
    cached_result = cache.get(
        request.question, request.top_k,
        request.filter_content_type, request.filter_doc_id,
        request.use_reranking,
    )
    
    if cached_result is not None:
        # Create a copy so we can mutate 'cached' safely
        response_dict = cached_result.dict()
        response_dict['cached'] = True
        return QueryApiResponse(**response_dict)

    qr = QueryRequest(
        question=request.question,
        top_k=request.top_k,
        filter_content_type=request.filter_content_type,
        filter_doc_id=request.filter_doc_id,
        use_reranking=request.use_reranking,
    )

    engine = get_generation_engine()
    result = engine.generate(query=None, request=qr)

    response = QueryApiResponse(
        answer=result.answer,
        sources=result.sources,
        citations=result.citations,
        confidence=result.confidence,
        retrieval_time_ms=result.retrieval_time_ms,
        generation_time_ms=result.generation_time_ms,
        total_time_ms=result.retrieval_time_ms + result.generation_time_ms,
        cached=False,
        llm_model=result.llm_model,
        transform_used=result.transform_used,
        reranker_used=result.reranker_used,
    )

    cache.set(
        request.question, request.top_k,
        request.filter_content_type, request.filter_doc_id,
        request.use_reranking, response,
    )

    return response
