"""
Error handling and request logging middleware.
"""

import time
import logging
import traceback
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catch-all exception handler for the API."""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            logger.error(f"Unhandled exception: {exc}")
            logger.error(traceback.format_exc())
            
            if isinstance(exc, ValueError):
                status_code = 400
                error_msg = "Bad Request"
            elif isinstance(exc, FileNotFoundError):
                status_code = 404
                error_msg = "Not Found"
            elif isinstance(exc, ConnectionError):
                status_code = 503
                error_msg = "Service Unavailable"
            elif isinstance(exc, PermissionError):
                status_code = 403
                error_msg = "Forbidden"
            else:
                status_code = 500
                error_msg = "Internal server error"
                
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": error_msg,
                    "detail": str(exc),
                    "status_code": status_code
                }
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log incoming requests and response times."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        process_time_ms = round(process_time * 1000)
        
        query_string = request.url.query
        path = request.url.path
        if query_string:
            path = f"{path}?{query_string}"
            
        logger.info(f"{request.method} {path} {response.status_code} {process_time_ms}ms")
        
        return response


def add_middleware(app) -> None:
    """Add all middleware to the FastAPI app."""
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
