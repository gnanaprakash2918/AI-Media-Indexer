"""API dependency injection components."""

from fastapi import Request

from core.ingestion.pipeline import IngestionPipeline



def get_pipeline(request: Request) -> IngestionPipeline | None:
    """Retrieve the ingestion pipeline from app state."""
    return getattr(request.app.state, "pipeline", None)


def get_search_agent(request: Request):
    """Retrieve the singleton SearchAgent from app state."""
    return getattr(request.app.state, "search_agent", None)
