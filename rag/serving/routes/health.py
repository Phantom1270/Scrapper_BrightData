"""
Health and status endpoints.
"""

import time
from fastapi import APIRouter

from rag.serving.schemas import HealthResponse
from rag.serving.dependencies import (
    get_start_time, get_store, get_index_builder
)


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check. Returns system status and component health.
    """
    start_time = get_start_time()
    uptime = time.time() - start_time if start_time else 0

    components = {}

    # Check storage
    try:
        store = get_store()
        count = store.count_documents()
        components["storage"] = {"status": "healthy", "documents": count}
    except Exception as e:
        components["storage"] = {"status": "unhealthy", "error": str(e)}

    # Check LLM
    try:
        from rag.llm import create_llm_client
        from rag.config.settings import get_settings
        settings = get_settings()
        llm = create_llm_client(settings)
        available = llm.is_available()
        components["llm"] = {
            "status": "healthy" if available else "unavailable",
            "model": llm.get_model_name(),
            "provider": settings.llm.provider,
        }
    except Exception as e:
        components["llm"] = {"status": "unavailable", "error": str(e)}

    # Check vector store
    try:
        builder = get_index_builder()
        vcount = builder.vector_store.count()
        components["vector_store"] = {"status": "healthy", "count": vcount}
    except Exception as e:
        components["vector_store"] = {"status": "unavailable", "error": str(e)}

    # Check BM25
    try:
        builder = get_index_builder()
        bcount = builder.bm25_index.count()
        components["bm25"] = {"status": "healthy", "count": bcount}
    except Exception as e:
        components["bm25"] = {"status": "unavailable", "error": str(e)}

    # Determine overall status
    statuses = [c.get("status", "unknown") for c in components.values()]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        uptime_seconds=round(uptime, 1),
        components=components,
    )


@router.get("/health/components")
async def component_health():
    """
    Detailed component health. Same as /health but returns
    just the components dict for quick checks.
    """
    health = await health_check()
    return health.components
