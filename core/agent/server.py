"""MCP server exposing media ingestion and search tools.

This module wraps the core ingestion and retrieval components of
AI-Media-Indexer as MCP tools so that LLM-based agents can call them
via the Model Context Protocol (MCP).

Exposed capabilities:

* Ingest a new video file into the vector database.
* Search the indexed media by natural language query.

The server is stateless at the protocol level but maintains shared
process-local singletons (VectorDB, SearchEngine, IngestionPipeline)
to avoid repeatedly bootstrapping heavy infrastructure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from core.ingestion.pipeline import IngestionPipeline
from core.retrieval.search import SearchEngine  # type: ignore[import]
from core.storage.db import VectorDB

mcp = FastMCP("AI Media Indexer")

# Singletons
_vector_db: VectorDB | None = None
_search_engine: SearchEngine | None = None
_pipeline: IngestionPipeline | None = None


def _get_vector_db() -> VectorDB:
    """Return a shared VectorDB instance, creating it on first use."""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDB(
            backend="docker",
            host="localhost",
            port=6333,
        )
    return _vector_db


def _get_search_engine() -> SearchEngine:
    """Return a shared SearchEngine instance, creating it on first use."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine(_get_vector_db())
    return _search_engine


def _get_pipeline() -> IngestionPipeline:
    """Return a shared IngestionPipeline instance, creating it on first use."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(
            qdrant_backend="docker",
            qdrant_host="localhost",
            qdrant_port=6333,
            frame_interval_seconds=15,
            # tmdb_api_key is taken from env by MetadataEngine if not passed.
            tmdb_api_key=None,
        )
    return _pipeline


class SearchInput(BaseModel):
    """Arguments for the media search tool."""

    query: str = Field(
        description="Natural language query (e.g. 'red car explosion at night').",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of results to return.",
    )


class SearchResponse(BaseModel):
    """Response payload for media search.

    Attributes:
        visual_matches: Frame-based search hits, typically containing file path,
            timestamp, similarity score, and a short description of the scene.
        dialogue_matches: Dialogue/text-based search hits, typically containing
            file path, timestamp, similarity score, and a transcript snippet.
    """

    visual_matches: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Frame-based/visual search matches.",
    )
    dialogue_matches: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Dialogue/text-based search matches.",
    )


class IngestInput(BaseModel):
    """Arguments for the media ingestion tool."""

    file_path: str = Field(
        description="Absolute path to the video file on the server filesystem.",
    )
    media_type: str = Field(
        default="unknown",
        description=(
            "Optional media type hint. Should match a MediaType value, e.g. "
            "'movie', 'tv', 'personal', or 'unknown'."
        ),
    )


class IngestResponse(BaseModel):
    """Response payload for media ingestion."""

    file_path: str = Field(
        description="Resolved absolute path of the ingested file.",
    )
    media_type_hint: str = Field(
        description="Media type hint that was forwarded to the pipeline.",
    )
    message: str = Field(
        description="Human-readable summary of the ingestion result.",
    )


# MCP tools


@mcp.tool()
async def search_media(args: SearchInput) -> SearchResponse:
    """Search the indexed media library for dialogue or visual matches.

    The concrete structure of each match entry is determined by the
    :class:`SearchEngine` implementation. Each list element typically
    contains a score, timestamp, file path, and content snippet.

    Args:
        args: Structured search arguments containing the query string
            and optional result limit.

    Returns:
        A :class:`SearchResponse` with separate lists for visual and
        dialogue matches.
    """
    engine = _get_search_engine()
    results = engine.search(args.query, limit=args.limit)
    return SearchResponse(**results)


@mcp.tool()
async def ingest_media(args: IngestInput) -> IngestResponse:
    """Ingest and index a single video file into the media database.

    This tool runs the full ingestion pipeline on the given file path,
    including metadata enrichment, subtitle/transcription processing,
    frame analysis, and face detection.

    Args:
        args: Structured ingestion arguments, including the absolute
            file path and an optional media type hint string.

    Returns:
        An :class:`IngestResponse` summarizing the ingestion outcome.

    Raises:
        FileNotFoundError: If the resolved path does not exist or is
            not a regular file.
    """
    raw_path = args.file_path.strip()
    path = Path(raw_path).expanduser().resolve()

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Media file not found or not a file: {path}")

    pipeline = _get_pipeline()
    await pipeline.process_video(path, media_type_hint=args.media_type)

    return IngestResponse(
        file_path=str(path),
        media_type_hint=args.media_type,
        message=f"Successfully indexed media file: {path}",
    )


def main() -> None:
    """Run the MCP server.

    This starts the FastMCP event loop and exposes the registered tools
    over the configured transport (e.g. stdio when launched as an MCP
    tool, or other transports depending on host configuration).
    """
    mcp.run()


if __name__ == "__main__":
    main()
