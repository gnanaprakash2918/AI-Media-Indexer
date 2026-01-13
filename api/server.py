"""FastAPI server composition root."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from uuid import uuid4

# Windows-specific asyncio fix for "WinError 10054" noise
if sys.platform == "win32":
    import asyncio
    import logging
    import warnings

    # Suppress pynvml unrelated warnings from torch/fastapi
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="torch.cuda"
    )

    # Custom filter to suppress ConnectionResetError from logging
    class ConnectionResetFilter(logging.Filter):
        """Filters out WinError 10054 and ConnectionResetError noise from logs.

        This is particularly useful on Windows where client disconnects
        can flood the server logs with expected socket errors.
        """

        def filter(self, record: logging.LogRecord) -> bool:
            """Determines if the record should be logged."""
            msg = str(record.getMessage())
            if "ConnectionResetError" in msg or "WinError 10054" in msg:
                return False
            if record.exc_info:
                exc_type = record.exc_info[0]
                if exc_type and issubclass(exc_type, ConnectionResetError):
                    return False
            return True

    # Apply filter to root logger and common loggers
    for logger_name in ["", "uvicorn", "uvicorn.error", "asyncio"]:
        logging.getLogger(logger_name).addFilter(ConnectionResetFilter())

    class SilenceEventLoopPolicy(asyncio.WindowsProactorEventLoopPolicy):
        """Custom event loop policy to suppress Windows-specific noise.

        Injects a custom exception handler into the event loop factory to
        gracefully ignore ConnectionResetError and WinError 10054.
        """

        def _loop_factory(self) -> asyncio.AbstractEventLoop:
            loop = super()._loop_factory()

            def exception_handler(loop, context):
                """Silently handles expected network-level errors on Windows."""
                # Suppress connection reset errors typical in Windows
                exc = context.get("exception")
                if exc:
                    if isinstance(exc, ConnectionResetError):
                        return
                    exc_str = str(exc)
                    if (
                        "ConnectionResetError" in exc_str
                        or "WinError 10054" in exc_str
                    ):
                        return
                # Default handler for everything else
                loop.default_exception_handler(context)

            loop.set_exception_handler(exception_handler)
            return loop

    asyncio.set_event_loop_policy(SilenceEventLoopPolicy())

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import routers
from api.routes import (
    agent,
    councils,
    identities,
    ingest,
    media,
    search,
    system,
)
from config import settings
from core.ingestion.jobs import job_manager
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import bind_context, clear_context, logger
from core.utils.observability import end_trace, init_langfuse, start_trace

pipeline: IngestionPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    init_langfuse()
    logger.info("startup")

    thumb_dir = settings.cache_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    global pipeline
    try:
        pipeline = IngestionPipeline()
        app.state.pipeline = pipeline
        logger.info("Pipeline initialized")

        # Crash Recovery
        recovery_stats = job_manager.recover_on_startup(timeout_seconds=60.0)
        if recovery_stats["paused"] > 0:
            logger.warning(
                f"Crash recovery: Marked {recovery_stats['paused']} interrupted jobs as PAUSED"
            )

    except Exception as exc:
        pipeline = None
        app.state.pipeline = None
        logger.error(f"Pipeline init failed: {exc}")

    yield

    if pipeline and pipeline.db:
        pipeline.db.close()
    logger.info("shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Media Indexer",
        version="2.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware: Observability
    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        """Injects trace IDs and records request-level observability metrics."""
        _ = request.headers.get("x-trace-id", str(uuid4()))
        bind_context(component="api")
        start_trace(
            name=request.url.path,
            metadata={
                "method": request.method,
                "client": request.client.host if request.client else None,
            },
        )
        try:
            response = await call_next(request)
            end_trace("success")
            return response
        except Exception as exc:
            end_trace("error", str(exc))
            raise
        finally:
            clear_context()

    # Mount Routers
    app.include_router(system.router, tags=["System"])
    app.include_router(media.router, tags=["Media"])
    app.include_router(ingest.router, tags=["Ingestion"])
    app.include_router(search.router, tags=["Search"])
    app.include_router(agent.router, tags=["Agent"])
    app.include_router(identities.router, tags=["Identities"])
    app.include_router(councils.router, tags=["Councils"])

    # Mount static files for default thumbnails/assets
    thumb_dir = settings.cache_dir / "thumbnails"
    app.mount(
        "/thumbnails", StaticFiles(directory=str(thumb_dir)), name="thumbnails"
    )

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:create_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        factory=True,
    )
