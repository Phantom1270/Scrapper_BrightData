"""
FastAPI application factory.
"""

from fastapi import FastAPI


def create_app(settings=None) -> FastAPI:
    """Create and configure the FastAPI application."""

    if settings is None:
        from rag.config.settings import get_settings
        settings = get_settings()

    app = FastAPI(
        title="Documentation RAG API",
        description="Ask questions about your documentation and get grounded, cited answers.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    from fastapi.middleware.cors import CORSMiddleware
    cors_origins = getattr(settings.serving, "cors_origins", ["http://localhost:3000", "http://localhost:5173"])
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from rag.serving.middleware import add_middleware
    add_middleware(app)

    from rag.serving.routes.query import router as query_router
    from rag.serving.routes.index import router as index_router
    from rag.serving.routes.evaluation import router as eval_router
    from rag.serving.routes.health import router as health_router

    app.include_router(query_router, prefix="/api/v1", tags=["query"])
    app.include_router(index_router, prefix="/api/v1", tags=["index"])
    app.include_router(eval_router, prefix="/api/v1", tags=["evaluation"])
    app.include_router(health_router, prefix="/api/v1", tags=["health"])

    from rag.serving.dependencies import initialize_dependencies, shutdown_dependencies

    @app.on_event("startup")
    async def startup():
        initialize_dependencies(settings)

    @app.on_event("shutdown")
    async def shutdown():
        shutdown_dependencies()

    return app
