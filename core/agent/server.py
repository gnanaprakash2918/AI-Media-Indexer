"""MCP server exposing media ingestion and search tools over stdio.

This module wraps the core AI-Media-Indexer functionality as MCP tools so
LLM-based agents can call them via the Model Context Protocol (MCP).

Exposed tools:

* `search_media`: Multimodal search across dialogue and visual frames.
* `ingest_media`: Full ingestion pipeline for a single video file.

The server is intended to be launched as a local MCP tool, typically via
`uv run python core/agent/server.py` or equivalent.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from core.retrieval.agentic_search import SearchAgent

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from core.ingestion.pipeline import IngestionPipeline
from core.retrieval.search import SearchEngine
from core.schemas import IngestResponse, SearchResponse
from core.storage.db import VectorDB
from core.utils.logger import log

# FORCE UTF-8 for stdout/stderr to prevent Windows charmap crashes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Initialize FastMCP server
mcp = FastMCP("MediaIndexer")

# Global singletons (lazy-loaded to avoid repeated heavy initialization)

_vector_db: VectorDB | None = None
_search_engine: SearchEngine | None = None
_pipeline: IngestionPipeline | None = None
_agentic_search: "SearchAgent | None" = None


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
            # MetadataEngine will fall back to env if set.
            tmdb_api_key=None,
        )
    return _pipeline


def _get_agentic_search() -> "SearchAgent":
    """Return a shared SearchAgent instance for FAANG-level search."""
    global _agentic_search
    if _agentic_search is None:
        from core.retrieval.agentic_search import SearchAgent
        _agentic_search = SearchAgent(db=_get_vector_db())
    assert _agentic_search is not None
    return _agentic_search


# MCP tools


@mcp.tool()
async def agentic_search(
    query: Annotated[
        str,
        Field(
            description="Natural language search query with entity support "
            "(e.g. 'Prakash bowling at Brunswick', 'someone eating idli').",
        ),
    ],
    limit: Annotated[
        int,
        Field(
            description="Maximum number of results to return.",
            ge=1,
            le=50,
        ),
    ] = 10,
    use_expansion: Annotated[
        bool, 
        Field(
            description="Use LLM to expand query (e.g. 'South Indian food' → 'idli, dosa').",
        ),
    ] = True,
) -> dict:
    """FAANG-level search with LLM query expansion and identity resolution.

    This tool uses an LLM to:
    1. Parse the query to extract person names, actions, objects, brands
    2. Expand cultural terms (e.g., 'South Indian breakfast' → 'idli, dosa')
    3. Resolve person names to face cluster IDs from HITL
    4. Filter frames by identity before vector search

    Args:
        query: Natural language query with optional entity references.
        limit: Maximum number of results.
        use_expansion: Whether to use LLM for query expansion.

    Returns:
        Dict with results, parsed query, and metadata.
    """
    agent = _get_agentic_search()
    return await agent.search(query, limit=limit, use_expansion=use_expansion)


@mcp.tool()
async def search_media(
    query: Annotated[
        str,
        Field(
            description="Natural language search query "
            "(e.g. 'red car explosion', 'argument in kitchen').",
        ),
    ],
    limit: Annotated[
        int,
        Field(
            description="Maximum number of results to return.",
            ge=1,
            le=50,
        ),
    ] = 5,
) -> SearchResponse:
    """Find relevant media segments (visual or dialogue) based on a query.

    This tool queries the underlying VectorDB via :class:`SearchEngine` and
    returns both frame-based and dialogue-based matches.

    Args:
        query: Natural language query string describing the desired scene or
            utterance.
        limit: Maximum number of results to return per modality.

    Returns:
        A :class:`SearchResponse` with ``visual_matches`` and
        ``dialogue_matches`` lists.
    """
    engine = _get_search_engine()
    results = engine.search(query, limit=limit)
    return SearchResponse(**results)


@mcp.tool()
async def ingest_media(
    file_path: Annotated[
        str,
        Field(
            description="Absolute path to the video file on the host filesystem.",
        ),
    ],
    media_type: Annotated[
        str,
        Field(
            description="Optional media type hint: 'movie', 'tv',"
            " 'personal', or 'unknown'.",
        ),
    ] = "unknown",
) -> IngestResponse:
    """Ingest and index a video file into the media library.

    This runs the full ingestion pipeline on the provided file path,
    including metadata enrichment, subtitle/transcription processing,
    frame analysis, and face detection.

    Args:
        file_path: Absolute path to the video file on disk.
        media_type: Optional media type hint string, forwarded to the
            ingestion pipeline for better classification.

    Returns:
        An :class:`IngestResponse` summarizing success or failure of the
        ingestion attempt.
    """
    # Normalize and resolve the path
    raw_path = file_path.strip().strip('"').strip("'")
    path = Path(raw_path).expanduser().resolve()

    if not path.exists() or not path.is_file():
        return IngestResponse(
            file_path=str(path),
            media_type_hint=media_type,
            message=f"Error: File not found or not a regular file: {path}",
        )

    pipeline = _get_pipeline()
    try:
        await pipeline.process_video(path, media_type_hint=media_type)
        msg = "Ingestion complete."
    except Exception as exc:  # noqa: BLE001
        msg = f"Ingestion failed: {exc}"

    return IngestResponse(
        file_path=str(path),
        media_type_hint=media_type,
        message=msg,
    )


def main() -> None:
    """Run the MCP server over stdio.

    This starts the FastMCP event loop and exposes the registered tools
    (`search_media`, `ingest_media`) to MCP-compatible hosts.
    """
    log("[MCP] MediaIndexer server started. Waiting for MCP client...")
    mcp.run()


if __name__ == "__main__":
    main()
