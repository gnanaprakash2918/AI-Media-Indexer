"""API dependency injection components."""

from fastapi import Request
from core.storage.db import VectorDB


def get_db(request: Request) -> VectorDB | None:
    """Retrieve the VectorDB instance from app state."""
    return getattr(request.app.state, "db", None)


def get_search_agent(request: Request):
    """Retrieve the singleton SearchAgent from app state."""
    return getattr(request.app.state, "search_agent", None)
