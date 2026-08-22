"""
Index management endpoints.
"""

import time
from fastapi import APIRouter
from rag.serving.schemas import IndexRequest, IndexStatusResponse
from rag.serving.dependencies import get_index_builder, get_store


router = APIRouter()


@router.post("/index")
async def trigger_index(request: IndexRequest):
    """
    Trigger indexing of all chunks.
    Builds vector store and BM25 index from stored chunks.
    """
    builder = get_index_builder()
    
    t0 = time.time()
    result = builder.build_all(force_rebuild=request.force_rebuild)
    elapsed = time.time() - t0
    
    return {
        "status": "completed",
        "total_chunks": result.total_chunks,
        "chunks_embedded": result.chunks_embedded,
        "embedding_model": result.embedding_model,
        "embedding_dimension": result.embedding_dimension,
        "vector_store_count": result.vector_store_count,
        "bm25_count": result.bm25_count,
        "skipped": result.skipped,
        "processing_time_seconds": round(elapsed, 2),
    }


@router.get("/index/status", response_model=IndexStatusResponse)
async def index_status():
    """
    Get the current status of the search indexes.
    """
    store = get_store()
    builder = get_index_builder()

    total_docs = store.count_documents()
    total_chunks = store.count_chunks()

    vector_count = 0
    bm25_count = 0
    
    try:
        vector_count = builder.vector_store.count()
    except Exception:
        pass
        
    try:
        bm25_count = builder.bm25_index.count()
    except Exception:
        pass

    if total_chunks == 0:
        status = "empty"
    elif vector_count == 0 and bm25_count == 0:
        status = "not_indexed"
    elif vector_count < total_chunks or bm25_count < total_chunks:
        status = "partial"
    else:
        status = "ready"

    return IndexStatusResponse(
        status=status,
        total_documents=total_docs,
        total_chunks=total_chunks,
        vector_store_count=vector_count,
        bm25_count=bm25_count,
    )
