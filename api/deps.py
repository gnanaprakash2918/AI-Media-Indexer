"""API dependency injection components."""

from fastapi import Request

from core.retrieval.query_pipeline import QueryPipeline


def get_pipeline(request: Request) -> QueryPipeline | None:
    """Retrieve the lightweight decoupled query pipeline from app state."""
    return getattr(request.app.state, "pipeline", None)


def get_search_agent(request: Request):
    """Retrieve the singleton SearchAgent from app state."""
    return getattr(request.app.state, "search_agent", None)
