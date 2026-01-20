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
# AND redirect stdout to stderr during imports to prevent Torch/PyNVML noise
# from corrupting the MCP JSON-RPC stream.
_original_stdout = sys.stdout
sys.stdout = sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer, encoding="utf-8", errors="replace"
)

# Initialize FastMCP server
mcp = FastMCP("MediaIndexer")

# Global singletons (lazy-loaded to avoid repeated heavy initialization)

_vector_db: VectorDB | None = None
_search_engine: SearchEngine | None = None
_pipeline: IngestionPipeline | None = None
_agentic_search: SearchAgent | None = None


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
            # Uses config.frame_interval (default 0.5s = 2fps) for accurate search
            tmdb_api_key=None,
        )
    return _pipeline


def _get_agentic_search() -> SearchAgent:
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
async def scenelet_search(
    query: Annotated[
        str,
        Field(
            description="Natural language query for an action sequences "
            "(e.g. 'person running then jumping', 'skateboarding trick').",
        ),
    ],
    limit: Annotated[
        int,
        Field(
            description="Maximum number of results.",
            ge=1,
            le=50,
        ),
    ] = 10,
    video_path: Annotated[
        str | None,
        Field(
            description="Optional absolute path to filter by video.",
        ),
    ] = None,
) -> dict:
    """Search for temporal action sequences (scenelets).

    This tool is best for finding specific activities or motions that happen
    over time, rather than static frames. It returns time ranges (start/end).

    Args:
        query: Action-based query string.
        limit: Max results.
        video_path: Optional video filter.

    Returns:
        Dict with scenelet results including start/end times.
    """
    agent = _get_agentic_search()
    return await agent.scenelet_search(
        query, limit=limit, video_path=video_path
    )


@mcp.tool()
async def query_video_rag(
    query: Annotated[
        str,
        Field(
            description="Natural language question or search query about video content "
            "(e.g. 'Why did he cry?', 'Find the red car scene').",
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
) -> dict:
    """Cognitive VideoRAG search with query decomposition and answer generation.

    This is the "High IQ" search that:
    1. Decomposes queries into visual/audio/identity components
    2. Searches across multiple modalities
    3. Generates text answers for questions with citations
    4. Uses external knowledge (Brave Search) when needed

    Args:
        query: Natural language query or question.
        limit: Maximum results to retrieve.

    Returns:
        Dict with structured query, results, and optional answer.
    """
    from core.retrieval.rag import get_orchestrator

    orchestrator = get_orchestrator()
    response = await orchestrator.search(query, limit=limit)
    return response.model_dump()


@mcp.tool()
async def get_video_summary(
    video_path: Annotated[
        str,
        Field(
            description="Path to the video file to summarize.",
        ),
    ],
    force_regenerate: Annotated[
        bool,
        Field(
            description="Force regenerating summaries even if they exist.",
        ),
    ] = False,
) -> dict:
    """Get or generate hierarchical summaries for a video.

    Returns two levels of summaries:
    - L1: Full video summary (plot, themes, main events)
    - L2: Scene summaries (5-minute chunks)

    Useful for answering "What is this video about?" questions.

    Args:
        video_path: Path to the video file.
        force_regenerate: If True, regenerate even if summaries exist.

    Returns:
        Dict with l1_summary, l2_summaries, and scene_count.
    """
    from core.processing.summarizer import summarizer

    return await summarizer.summarize_video(video_path, force=force_regenerate)


@mcp.tool()
async def enrich_identity(
    context: Annotated[
        str,
        Field(
            description="Context about the identity to enrich "
            "(e.g. 'person in blue shirt at bowling alley').",
        ),
    ],
    entity_type: Annotated[
        str,
        Field(
            description="Type of entity: 'face', 'location', or 'topic'.",
        ),
    ] = "face",
) -> dict:
    """Enrich an unknown identity using external knowledge (Brave Search).

    This tool queries the web to identify unknown people, locations, or topics.
    Useful when a face is detected but not recognized.

    Note: Requires BRAVE_API_KEY to be configured. Returns empty results otherwise.

    Args:
        context: Contextual description of the entity.
        entity_type: Type of entity to enrich.

    Returns:
        Dict with possible_matches and confidence.
    """
    from core.processing.enrichment import enricher

    if entity_type == "face":
        return await enricher.enrich_unknown_face(context=context)
    elif entity_type == "location":
        return await enricher.enrich_location(location_hint=context)
    else:
        return await enricher.enrich_topic(topic=context)


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
    except Exception as exc:
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
    # Restore stdout for MCP communication
    sys.stdout = io.TextIOWrapper(
        _original_stdout.buffer, encoding="utf-8", errors="replace"
    )
    log("[MCP] MediaIndexer server started. Waiting for MCP client...")
    mcp.run()


if __name__ == "__main__":
    main()
