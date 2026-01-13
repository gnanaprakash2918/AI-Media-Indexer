"""API dependency injection components."""

from fastapi import Request

from core.ingestion.pipeline import IngestionPipeline


def get_pipeline(request: Request) -> IngestionPipeline | None:
    """Retrieve the ingestion pipeline from app state."""
    return getattr(request.app.state, "pipeline", None)
