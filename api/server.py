"""FastAPI server configuration and routes."""

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import bind_context, clear_context, logger
from core.utils.observability import end_trace, init_langfuse, start_trace

# Global Pipeline
pipeline: IngestionPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_langfuse()
    logger.info("startup")

    thumb_dir = settings.cache_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    global pipeline
    try:
        pipeline = IngestionPipeline()
        logger.info("Pipeline initialized")
    except Exception as exc:
        pipeline = None
        logger.error(f"Pipeline init failed: {exc}")

    yield

    if pipeline and pipeline.db:
        pipeline.db.close()
    logger.info("shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Media Indexer",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
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

    # Routes

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "device": settings.device,
            "pipeline": "ready" if pipeline else "unavailable",
            "qdrant": "connected" if pipeline and pipeline.db else "disconnected",
        }


    # Input Schema
    class IngestRequest(BaseModel):
        path: str
        media_type_hint: str = "unknown"

    @app.post("/ingest")
    async def ingest_media(request: IngestRequest, background_tasks: BackgroundTasks):
        """Trigger processing of a local file in the background."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        file_path = Path(request.path)
        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"File not found on server: {request.path}"
            )

        # Run the heavy pipeline in background
        background_tasks.add_task(
            pipeline.process_video, file_path, request.media_type_hint
        )

        return {
            "status": "queued",
            "file": str(file_path),
            "message": "Processing started in background. Check logs or poll /search."
        }

    @app.get("/search")
    async def search(q: str, limit: int = 20):
        """Semantic Search across Audio, Visual, and Voice."""
        if not pipeline:
            return {"error": "Pipeline not initialized"}

        # This searches Qdrant
        results = pipeline.db.search_media(q, limit=limit)
        return results

    # Serve Thumbnails
    thumb_path = settings.cache_dir / "thumbnails"
    thumb_path.mkdir(parents=True, exist_ok=True)
    app.mount("/thumbnails", StaticFiles(directory=str(thumb_path)), name="thumbnails")

    return app

app = create_app()
