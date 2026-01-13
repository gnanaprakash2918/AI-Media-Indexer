from typing import Optional

from fastapi import Request

from core.ingestion.pipeline import IngestionPipeline


def get_pipeline(request: Request) -> Optional[IngestionPipeline]:
    """Retrieve the ingestion pipeline from app state."""
    return getattr(request.app.state, "pipeline", None)
