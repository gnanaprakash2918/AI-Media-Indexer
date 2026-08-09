"""Lightweight query pipeline for the Web/API server.

This decoupled structure prevents the FastAPI web process from inheriting massive
ingestion dependencies (Whisper, PaddleOCR, PyAnnote, OpenCV, etc.) yielding
strict VRAM bounding and memory isolation as requested in MVP Phase 5.
"""

from core.storage.db import VectorDB
from core.utils.logger import log


class QueryPipeline:
    """Provides essential DB connectivity for search execution paths."""

    def __init__(self):
        log("[QueryPipeline] Initializing decoupled lightweight search core...")
        self.db = VectorDB()
        self.graph_builder = None
