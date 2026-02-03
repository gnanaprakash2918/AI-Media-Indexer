"""FastAPI server composition root."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import warnings
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Configure TensorFlow BEFORE any imports that trigger TF loading
print("DEBUG: Starting detailed server imports...")
print("DEBUG: Importing os/warnings/logging...")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL", "2"
)  # Suppress C++ level warnings

# Suppress tf_keras deprecation warnings BEFORE importing anything that uses TF
# These come through Python's warnings module and absl logging
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=".*tensorflow.*"
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=".*tf_keras.*"
)
warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*")
warnings.filterwarnings("ignore", message=".*deprecated.*", module=".*keras.*")

# Also filter via logging (tf_keras uses this path)
print("DEBUG: Configuring loggers...")
for logger_name in ["tensorflow", "tf_keras", "absl"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Windows-specific asyncio fix for "WinError 10054" noise
if sys.platform == "win32":
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

print("DEBUG: Importing FastAPI...")
# Imports already at top

# Import routers
print("DEBUG: Importing API routers...")
from api.routes import (  # noqa: E402
    agent,
    councils,
    events,
    faces,
    graph,
    identities,
    ingest,
    library,
    manipulation,
    media,
    search,
    system,
    tracking,  # [NEW] SAM 3 Object Tracking
    voices,
)

print("DEBUG: Checking overlays...")
try:
    from api.routes import overlays  # noqa: E402
except ImportError:
    overlays = None
print("DEBUG: Importing config & pipeline...")
from config import settings  # noqa: E402
from core.ingestion.jobs import job_manager  # noqa: E402
from core.ingestion.pipeline import IngestionPipeline  # noqa: E402
from core.utils.logger import bind_context, clear_context, logger  # noqa: E402
from core.utils.model_warmer import warmup_models  # [NEW] Warmer
from core.utils.observability import (  # noqa: E402
    end_trace,
    init_langfuse,
    start_trace,
)

pipeline: IngestionPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    init_langfuse()
    logger.info("startup")

    thumb_dir = settings.cache_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # Warmup Models (Background but awaited for safety or fire-and-forget?)
    # "Download needed ones not fallbacks at starting itself" -> Await it.
    try:
        await warmup_models()
    except Exception as e:
        logger.warning(f"Model warmup warning: {e}")

    global pipeline
    try:
        print("DEBUG: Lifespan: Initializing pipeline...")  # Added debug
        pipeline = IngestionPipeline()
        app.state.pipeline = pipeline
        logger.info("Pipeline initialized")

        # Crash Recovery
        recovery_stats = job_manager.recover_on_startup(timeout_seconds=60.0)
        if recovery_stats["paused"] > 0:
            logger.warning(
                f"Crash recovery: Marked {recovery_stats['paused']} interrupted jobs as PAUSED"
            )

        # Initialize Search Agent (Singleton)
        try:
            from core.retrieval.agentic_search import SearchAgent

            # Use the DB from the pipeline
            app.state.search_agent = SearchAgent(pipeline.db)
            logger.info("SearchAgent initialized (Singleton)")
        except Exception as sa_err:
            logger.error(f"SearchAgent init failed: {sa_err}")
            app.state.search_agent = None

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
    print("DEBUG: create_app() start...")
    app = FastAPI(
        title="AI Media Indexer",
        version="2.1.0",
        lifespan=lifespan,
    )

    # CORS: Restrict to known origins (override via CORS_ORIGINS env var)
    allowed_origins = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
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
    print("DEBUG: Mounting routers...")
    app.include_router(system.router, tags=["System"])
    app.include_router(media.router, tags=["Media"])
    app.include_router(ingest.router, tags=["Ingestion"])
    app.include_router(search.router, tags=["Search"])
    app.include_router(agent.router, tags=["Agent"])
    app.include_router(identities.router, tags=["Identities"])
    app.include_router(councils.router, tags=["Councils"])
    app.include_router(events.router, tags=["Events"])
    app.include_router(faces.router, tags=["Faces"])
    app.include_router(voices.router, tags=["Voices"])
    app.include_router(library.router, tags=["Library"])
    app.include_router(library.router, tags=["Library"])
    app.include_router(graph.router, tags=["Graph"])  # New
    app.include_router(tracking.router, tags=["Tracking"])  # [NEW] SAM 3
    if overlays:
        app.include_router(overlays.router, tags=["Overlays"])

    # Manipulation Routes
    app.include_router(manipulation.router, tags=["Manipulation"])

    # Mount static files for default thumbnails/assets
    thumb_dir = settings.cache_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)  # Create dir if not exists
    app.mount(
        "/thumbnails", StaticFiles(directory=str(thumb_dir)), name="thumbnails"
    )

    print("DEBUG: create_app() done!")
    return app


# Create module-level app for "uvicorn api.server:app" imports
print("DEBUG: Creating module-level app...")
app = create_app()
print("DEBUG: App created!")


if __name__ == "__main__":
    import uvicorn

    print("DEBUG: Starting Uvicorn run...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
    print("DEBUG: Uvicorn run exited.")
