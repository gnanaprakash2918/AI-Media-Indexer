"""API routes for search operations."""

import time
from collections import defaultdict
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query

# Clean dependency injection
from api.deps import get_pipeline
from api.schemas import AdvancedSearchRequest
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import logger

# Check for import ability or handle graceful degradation without lazy hacks if possible.
# Ideally SearchAgent is stable.
try:
    from core.retrieval.agentic_search import SearchAgent
except ImportError:
    SearchAgent = None
    logger.warning(
        "SearchAgent could not be imported. Agentic search will be disabled."
    )

router = APIRouter()


def _normalize_results(results: list[dict]) -> list[dict]:
    """Normalize search results to ensure consistent frontend fields.

    Ensures all results have video_path, timestamp, and thumbnail_url.
    """
    normalized = []
    for r in results:
        # Normalize video path (some collections use media_path)
        video = r.get("video_path") or r.get("media_path", "")
        ts = r.get("timestamp") or r.get("start_time") or r.get("start", 0)

        result = {
            **r,
            "video_path": video,
            "timestamp": ts,
        }

        # Add thumbnail_url if not present
        if video and "thumbnail_url" not in result:
            safe_path = quote(str(video))
            result["thumbnail_url"] = f"/media/thumbnail?path={safe_path}&time={ts}"
            result["playback_url"] = f"/media?path={safe_path}#t={max(0, ts - 3)}"

        normalized.append(result)
    return normalized

@router.get("/search/hybrid")
async def hybrid_search(
    q: Annotated[str, Query(..., description="Search query")],
    limit: Annotated[int, Query(description="Maximum results")] = 20,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)] = None,
    video_path: Annotated[
        str | None, Query(description="Optional video filter")
    ] = None,
    use_reranking: Annotated[
        bool, Query(description="Use LLM re-ranking for higher accuracy")
    ] = True,
    face_cluster_id: Annotated[
        int | None, Query(description="Filter by specific face cluster ID")
    ] = None,
) -> dict:
    """Performs a SOTA hybrid search with identity resolution and re-ranking.

    Combines vector similarity, keyword matching, and LLM-based verification
    to find the most relevant video segments.

    Args:
        q: The natural language search query.
        limit: Maximum number of results to return.
        pipeline: The core ingestion pipeline instance.
        video_path: Optional path to filter results by a specific video.
        use_reranking: Whether to enable the second-stage LLM verification.
        face_cluster_id: Optional filter for a specific face cluster.

    Returns:
        A dictionary containing ranked results, parsed query info, and stats.

    Raises:
        HTTPException: If the pipeline or database is uninitialized.
    """
    start_time = time.perf_counter()
    logger.info(f"[Search] Hybrid search: '{q}' (reranking: {use_reranking})")

    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        if SearchAgent:
            # Re-instantiate per request or usage? SearchAgent seems lightweight (just helper)
            agent = SearchAgent(db=pipeline.db)
            result = await agent.sota_search(
                query=q,
                limit=limit,
                video_path=video_path,
                use_reranking=use_reranking,
            )

            # Transform results for frontend compatibility
            transformed_results = []
            for r in result.get("results", []):
                # Normalize video_path (some collections use media_path)
                video = r.get("video_path") or r.get("media_path", "")
                ts = (
                    r.get("timestamp")
                    or r.get("start_time")
                    or r.get("start", 0)
                )
                transformed = {
                    **r,
                    "video_path": video,  # Ensure video_path is always present
                    "match_reason": r.get(
                        "llm_reasoning", r.get("reasoning", "")
                    ),
                    "agent_thought": r.get("llm_reasoning", ""),
                    "matched_constraints": [
                        c.get("constraint", "")
                        for c in r.get("constraints_satisfied", [])
                    ]
                    if r.get("constraints_satisfied")
                    else [],
                    "score": r.get("combined_score", r.get("score", 0.5)),
                    "timestamp": ts,
                }
                # Add thumbnail_url if not present
                if video and "thumbnail_url" not in transformed:
                    safe_path = quote(str(video))
                    transformed["thumbnail_url"] = (
                        f"/media/thumbnail?path={safe_path}&time={ts}"
                    )
                    transformed["playback_url"] = (
                        f"/media?path={safe_path}#t={max(0, ts - 3)}"
                    )
                transformed_results.append(transformed)

            duration = time.perf_counter() - start_time
            logger.info(
                f"[Search] SOTA search completed in {duration:.2f}s with {len(transformed_results)} results"
            )

            return {
                "query": q,
                "video_filter": video_path,
                "results": transformed_results,
                "stats": {
                    "total": len(transformed_results),
                    "duration_seconds": duration,
                },
                "parsed": result.get("parsed", {}),
                "search_type": "sota",
                # Debug/Transparency fields for frontend panel
                "pipeline_steps": result.get("pipeline_steps", []),
                "reasoning_chain": result.get("reasoning_chain", {}),
                "search_text": result.get("search_text", q),
                "fallback_used": result.get("fallback_used"),
            }
        else:
            # Fallback to Hybrid Search (Vector + Text)
            logger.warning(
                "[Search] SearchAgent unavailable, using hybrid search"
            )
            # Use hybrid search instead of pure vector search for better accuracy
            results = _normalize_results(
                pipeline.db.search_frames_hybrid(query=q, limit=limit)
            )
            duration = time.perf_counter() - start_time
            return {
                "query": q,
                "video_filter": video_path,
                "results": results,
                "stats": {
                    "total": len(results),
                    "duration_seconds": duration,
                },
                "search_type": "basic_fallback",
            }

    except Exception as e:
        logger.error(f"[Search] Hybrid search failed: {e}")
        try:
            # Fallback to hybrid search
            results = _normalize_results(
                pipeline.db.search_frames_hybrid(query=q, limit=limit)
            )
            return {
                "query": q,
                "results": results,
                "stats": {"total": len(results), "duration_seconds": 0},
                "error": str(e),
                "search_type": "error_fallback",
            }
        except Exception as fallback_err:
            raise HTTPException(
                status_code=500, detail=f"Search failed: {e}"
            ) from fallback_err


@router.post("/search/hybrid")
async def hybrid_search_post(
    request: AdvancedSearchRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> dict:
    """POST endpoint for hybrid search, supporting complex structured queries.

    Args:
        request: The search request payload with filters and settings.
        pipeline: The core ingestion pipeline instance.

    Returns:
        A dictionary of ranked search results.
    """
    return await hybrid_search(
        q=request.query,
        video_path=request.video_path,
        limit=request.limit,
        use_reranking=request.use_rerank,
        pipeline=pipeline,
    )


@router.post("/search/advanced")
async def advanced_search(
    req: AdvancedSearchRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Executes an advanced search with filtering and reranking."""
    from core.retrieval.engine import get_search_engine

    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    engine = get_search_engine(db=pipeline.db)
    results = await engine.search(
        query=req.query,
        use_rerank=req.use_rerank,
        limit=req.limit,
    )

    filtered = [r for r in results if r.score >= req.min_confidence]

    return {
        "query": req.query,
        "total": len(filtered),
        "results": [r.model_dump() for r in filtered],
    }


@router.get("/search")
async def search(
    q: str,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    limit: int = 20,
    search_type: Annotated[
        str,
        Query(
            description="Type of search: all, dialogue, visual, voice",
        ),
    ] = "all",
    video_path: Annotated[
        str | None,
        Query(
            description="Filter results to specific video path",
        ),
    ] = None,
) -> dict:
    """Performs a multi-modal semantic search with deduplication.

    Args:
        q: The search query string.
        limit: Maximum results to return per modality.
        search_type: Filter by 'dialogue', 'visual', 'voice', or 'all'.
        video_path: Optional absolute path filter for a specific video.
        pipeline: The core ingestion pipeline instance.

    Returns:
        A dictionary containing aggregated results and scoring statistics.
    """
    if not pipeline:
        return {"error": "Pipeline not initialized", "results": [], "stats": {}}

    start_time_search = time.perf_counter()
    logger.info(
        f"Search query: '{q}' | type: {search_type} | limit: {limit} | video_filter: {video_path}"
    )

    results = []
    stats = {
        "query": q,
        "search_type": search_type,
        "video_filter": video_path,
        "dialogue_count": 0,
        "visual_count": 0,
        "total_frames_scanned": 0,
    }

    # Dialogue/transcript search
    if search_type in ("all", "dialogue"):
        dialogue_results = pipeline.db.search_media(q, limit=limit * 2)

        # Post-filter by video_path
        if video_path:
            dialogue_results = [
                r for r in dialogue_results if r.get("video_path") == video_path
            ]

        stats["dialogue_count"] = len(dialogue_results)

        for hit in dialogue_results[:limit]:
            hit["result_type"] = "dialogue"
            hit["thumbnail_url"] = None
            video = hit.get("video_path")
            start_time = hit.get("start", 0)
            if video:
                safe_path = quote(str(video))
                hit["thumbnail_url"] = (
                    f"/media/thumbnail?path={safe_path}&time={start_time}"
                )
                hit["playback_url"] = f"/media?path={safe_path}#t={start_time}"

        results.extend(dialogue_results[:limit])
        logger.info(f"  Dialogue results: {len(dialogue_results)}")

    # Visual/frame search
    if search_type in ("all", "visual"):
        # Fetch more results to allow for deduplication and filtering
        frame_results = pipeline.db.search_frames(q, limit=limit * 4)

        # Post-filter by video_path
        if video_path:
            frame_results = [
                r for r in frame_results if r.get("video_path") == video_path
            ]

        stats["total_frames_scanned"] = len(frame_results)

        # Group by video and deduplicate
        unique_frames = []
        seen_timestamps: dict[str, list[float]] = {}
        occurrence_tracker: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "timestamps": []}
        )

        for hit in frame_results:
            video = hit.get("video_path")
            ts = hit.get("timestamp", 0)

            # Track all occurrences for stats
            if video:
                occurrence_tracker[video]["count"] += 1
                occurrence_tracker[video]["timestamps"].append(ts)

            # Deduplication: Skip if within 5 seconds
            if video in seen_timestamps:
                if any(
                    abs(ts - existing) < 5.0
                    for existing in seen_timestamps[video]
                ):
                    continue
            elif video is not None:
                seen_timestamps[video] = []

            if video is not None:
                seen_timestamps[video].append(ts)

            # Expand to 7-second context
            start_context = max(0.0, ts - 3.5)
            end_context = ts + 3.5

            hit["result_type"] = "visual"
            hit["start"] = start_context
            hit["end"] = end_context
            hit["original_timestamp"] = ts

            if video:
                safe_path = quote(str(video))
                hit["thumbnail_url"] = (
                    f"/media/thumbnail?path={safe_path}&time={ts}"
                )
                hit["playback_url"] = (
                    f"/media?path={safe_path}#t={start_context}"
                )
                hit["segment_url"] = (
                    f"/media/segment?path={safe_path}&start={start_context:.2f}&end={end_context:.2f}"
                )

            unique_frames.append(hit)
            if len(unique_frames) >= limit:
                break

        # Add occurrence stats
        for hit in unique_frames:
            video = hit.get("video_path")
            if video and video in occurrence_tracker:
                hit["occurrence_count"] = occurrence_tracker[video]["count"]
                hit["occurrence_timestamps"] = sorted(
                    occurrence_tracker[video]["timestamps"]
                )[:10]

        stats["visual_count"] = len(unique_frames)
        results.extend(unique_frames)
        logger.info(
            f"  Unique visual results: {len(unique_frames)} (from {len(frame_results)} raw)"
        )

    # Sort by score
    results.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
    final_results = results[:limit]

    stats["returned_count"] = len(final_results)

    if final_results:
        scores = [float(r.get("score", 0)) for r in final_results]
        stats["score_range"] = {
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores),
        }
        duration = time.perf_counter() - start_time_search
        logger.info(
            f"  Search complete: {len(final_results)} results | Score={min(scores):.4f}-{max(scores):.4f} | Duration={duration:.3f}s"
        )

    if not final_results:
        logger.warning(f"No results found for query: '{q}'. Using fallback.")
        fallback_results = pipeline.db.get_recent_frames(limit=10)
        for hit in fallback_results:
            video = hit.get("video_path")
            ts = hit.get("timestamp", 0)
            if video:
                safe_path = quote(str(video))
                hit["thumbnail_url"] = (
                    f"/media/thumbnail?path={safe_path}&time={ts}"
                )
                hit["playback_url"] = (
                    f"/media?path={safe_path}#t={max(0, ts - 3)}"
                )

        final_results = fallback_results
        stats["fallback"] = True
        stats["message"] = (
            "No exact matches found. Showing recent indexed content."
        )

    return {
        "results": final_results,
        "stats": stats,
    }


@router.get("/search/agentic")
async def agentic_search(
    q: str,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    limit: int = 20,
    use_expansion: bool = True,
):
    """FAANG-level search with LLM query expansion and identity resolution."""
    if not pipeline:
        return {"error": "Pipeline not initialized", "results": []}

    try:
        # Use simple import if reliable, otherwise fallback handled at top level
        if SearchAgent:
            agent = SearchAgent(db=pipeline.db)
            result = await agent.search(
                q, limit=limit, use_expansion=use_expansion
            )
            return result
        else:
            raise ImportError("SearchAgent undefined")

    except Exception as e:
        logger.error(f"Agentic search failed: {e}")
        # Fallback to hybrid search
        regular_results = pipeline.db.search_frames_hybrid(query=q, limit=limit)
        return {
            "query": q,
            "parsed": None,
            "error": str(e),
            "fallback": True,
            "results": regular_results,
        }


@router.get("/search/scenes")
async def scene_search(
    q: str,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    limit: int = 20,
    use_expansion: bool = True,
    video_path: Annotated[
        str | None, Query(description="Filter to specific video")
    ] = None,
):
    """Production-grade scene-level search for complex queries."""
    if not pipeline:
        return {"error": "Pipeline not initialized", "results": []}

    start_time_search = time.perf_counter()
    logger.info(
        f"[SceneSearch] Query: '{q[:100]}...' | video_filter: {video_path}"
    )

    try:
        if SearchAgent:
            agent = SearchAgent(db=pipeline.db)
            result = await agent.search_scenes(
                query=q,
                limit=limit,
                use_expansion=use_expansion,
                video_path=video_path,
            )

            for hit in result.get("results", []):
                video = hit.get("media_path")
                start = hit.get("start_time", 0)
                if video:
                    safe_path = quote(str(video))
                    hit["thumbnail_url"] = (
                        f"/media/thumbnail?path={safe_path}&time={start}"
                    )
                    hit["playback_url"] = f"/media?path={safe_path}#t={start}"

            duration = time.perf_counter() - start_time_search
            result["stats"] = {
                "duration_seconds": duration,
                "search_mode": "scene",
            }
            logger.info(
                f"[SceneSearch] Returned {len(result.get('results', []))} scenes in {duration:.3f}s"
            )
            return result
        else:
            raise ImportError("SearchAgent undefined")

    except Exception as e:
        logger.error(f"[SceneSearch] Error: {e}")
        return {
            "query": q,
            "error": str(e),
            "fallback": True,
            "results": pipeline.db.search_frames_hybrid(query=q, limit=limit),
        }


@router.post("/search/granular")
async def granular_search(
    query: str,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    limit: int = 20,
    video_path: Annotated[
        str | None, Query(description="Filter to specific video")
    ] = None,
    show_reasoning: bool = True,
    enable_rerank: bool = True,
):
    """Ultra-fine-grained search for complex 200+ word queries.

    Uses MultiVectorSearcher with:
    - LLM query decomposition
    - Identity resolution (names → cluster IDs)
    - Hybrid search (vector + keyword + RRF)
    - LLM reranking with chain-of-thought
    - Constraint scoring and filtering

    Args:
        query: Complex natural language query (up to 200+ words).
        pipeline: The core ingestion pipeline instance.
        limit: Maximum results to return.
        video_path: Optional filter to specific video.
        show_reasoning: Include reasoning trace in response.
        enable_rerank: Enable LLM-based reranking (slower but more accurate).

    Returns:
        Results with matched constraints, reasoning, and confidence.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    start_time_search = time.perf_counter()
    word_count = len(query.split())
    logger.info(
        f"[GranularSearch] Query: {word_count} words | video: {video_path}"
    )

    try:
        # Use HyperGranularSearcher (formerly unused dead code)
        try:
            from core.retrieval.hyper_granular_search import HyperGranularSearcher
            searcher = HyperGranularSearcher(db=pipeline.db)
            logger.info("[GranularSearch] Using HyperGranularSearcher")
        except Exception as e:
            logger.warning(f"HyperGranularSearcher init failed ({e}), falling back to MultiVector")
            from core.retrieval.advanced_query import MultiVectorSearcher
            searcher = MultiVectorSearcher(db=pipeline.db)

        # HyperGranularSearcher signature: search(query, limit, video_path)
        # MultiVectorSearcher signature: search(query, limit, video_path, enable_rerank)
        if hasattr(searcher, "decompose_query"): # It's HyperGranular
             results = await searcher.search(
                query=query,
                limit=limit,
                video_path=video_path,
            )
        else:
            results = await searcher.search(
                query=query,
                limit=limit,
                video_path=video_path,
                enable_rerank=enable_rerank,
            )

        # Add thumbnail/playback URLs
        for hit in results:
            video = hit.get("video_path")
            ts = hit.get("timestamp", hit.get("start_time", 0))
            if video:
                safe_path = quote(str(video))
                hit["thumbnail_url"] = (
                    f"/media/thumbnail?path={safe_path}&time={ts}"
                )
                hit["playback_url"] = (
                    f"/media?path={safe_path}#t={max(0, ts - 3)}"
                )

        duration = time.perf_counter() - start_time_search

        # Build response
        # Build pipeline_steps for debug panel
        pipeline_steps = [
            {
                "step": "Query Decomposition",
                "status": "completed",
                "detail": f"Parsed {len(results[0].get('decomposed_constraints', []))} constraints"
                if results
                else "No constraints",
                "data": {
                    "word_count": word_count,
                    "modalities": results[0].get("query_modalities", [])
                    if results
                    else [],
                },
            },
            {
                "step": "Multi-Vector Search",
                "status": "completed",
                "detail": f"Found {len(results)} candidates",
                "data": {
                    "collection": "media_frames",
                    "enable_rerank": enable_rerank,
                },
            },
            {
                "step": "LLM Reranking",
                "status": "completed" if enable_rerank else "skipped",
                "detail": f"{'Applied' if enable_rerank else 'Disabled'} → {len(results)} final results",
                "data": {
                    "enabled": enable_rerank,
                    "final_count": len(results),
                },
            },
        ]

        response = {
            "query": query,
            "query_word_count": word_count,
            "video_filter": video_path,
            "results": results,
            "stats": {
                "total": len(results),
                "duration_seconds": round(duration, 3),
                "constraints_parsed": len(
                    results[0].get("decomposed_constraints", [])
                )
                if results
                else 0,
                "reranking_enabled": enable_rerank,
            },
            # Debug/Transparency fields
            "pipeline_steps": pipeline_steps,
            "search_text": query,
        }

        if show_reasoning and results:
            response["reasoning_trace"] = results[0].get("reasoning_trace", [])
            response["decomposed_constraints"] = results[0].get(
                "decomposed_constraints", []
            )
            response["query_modalities"] = results[0].get(
                "query_modalities", []
            )

        logger.info(
            f"[GranularSearch] Returned {len(results)} results in {duration:.3f}s"
        )
        return response

    except ImportError as e:
        logger.warning(f"[GranularSearch] Import error: {e}, using fallback")
        results = pipeline.db.search_frames(query=query, limit=limit)
        return {
            "query": query,
            "results": results,
            "stats": {"total": len(results), "fallback": True},
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"[GranularSearch] Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Granular search failed: {e}"
        ) from e


@router.get("/search/explainable")
async def explainable_search(
    q: Annotated[str, Query(..., description="Search query")],
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    limit: Annotated[int, Query(description="Maximum results")] = 10,
) -> dict:
    """Search with detailed reasoning for each result.

    Returns explainable matches with:
    - Matched entities with individual confidence scores
    - Reasoning for why each result was selected
    - Face/voice identification with names
    - Evidence types (face_match, voice_match, text_match, etc.)

    Best for debugging and understanding search quality.

    Args:
        q: Natural language search query.
        pipeline: The core ingestion pipeline instance.
        limit: Maximum number of results to return.

    Returns:
        Results with explainable metadata and reasoning.
    """
    start_time = time.perf_counter()
    logger.info(f"[ExplainableSearch] Query: '{q[:80]}...'")

    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Parse query for entity extraction
        parsed = None
        if SearchAgent:
            agent = SearchAgent(db=pipeline.db)
            parsed = await agent.parse_query(q)

        # Call the explainable search method
        results = pipeline.db.explainable_search(
            query_text=q,
            parsed_query=parsed,
            limit=limit,
        )

        # Add thumbnail URLs
        for r in results:
            video = r.get("video_path") or r.get("media_path", "")
            ts = r.get("timestamp", r.get("start_time", 0))
            if video and "thumbnail_url" not in r:
                safe_path = quote(str(video))
                r["thumbnail_url"] = f"/media/thumbnail?path={safe_path}&time={ts}"
                r["playback_url"] = f"/media?path={safe_path}#t={max(0, ts - 3)}"

        duration = time.perf_counter() - start_time
        logger.info(
            f"[ExplainableSearch] Returned {len(results)} results in {duration:.3f}s"
        )

        return {
            "query": q,
            "results": results,
            "stats": {
                "total": len(results),
                "duration_seconds": round(duration, 3),
            },
            "search_type": "explainable",
            "parsed_query": parsed.model_dump() if parsed else None,
        }

    except Exception as e:
        logger.error(f"[ExplainableSearch] Error: {e}")
        # Fallback to hybrid search
        results = _normalize_results(
            pipeline.db.search_frames_hybrid(query=q, limit=limit)
        )
        return {
            "query": q,
            "results": results,
            "stats": {"total": len(results), "fallback": True},
            "error": str(e),
            "search_type": "fallback",
        }

@router.get("/search/multimodal")
async def multimodal_search(
    q: Annotated[str, Query(..., description="Search query")],
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    limit: Annotated[int, Query(description="Maximum results")] = 20,
    video_path: Annotated[
        str | None, Query(description="Filter to specific video")
    ] = None,
    use_reranking: Annotated[
        bool, Query(description="Use LLM re-ranking")
    ] = True,
) -> dict:
    """Comprehensive multimodal search using ALL indexed data sources.

    Fuses results from:
    - Scenes (visual + motion + dialogue vectors)
    - Voice segments (speaker diarization)
    - Face clusters (identity)
    - Co-occurrences (temporal relationships)

    Uses Reciprocal Rank Fusion (RRF) to merge modalities.

    Args:
        q: Natural language search query.
        pipeline: The core ingestion pipeline instance.
        limit: Maximum number of results to return.
        video_path: Optional filter for a specific video.
        use_reranking: Whether to enable LLM verification.

    Returns:
        Fused results with modality breakdown and reasoning.
    """
    start_time_search = time.perf_counter()
    logger.info(f"[MultimodalSearch] Query: '{q[:80]}...'")

    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        if SearchAgent:
            agent = SearchAgent(db=pipeline.db)
            result = await agent.comprehensive_multimodal_search(
                query=q,
                limit=limit,
                video_path=video_path,
                use_reranking=use_reranking,
            )

            # Add thumbnail/playback URLs
            for hit in result.get("results", []):
                video = hit.get("video_path") or hit.get("media_path", "")
                ts = hit.get("start_time", hit.get("timestamp", 0))
                if video and "thumbnail_url" not in hit:
                    safe_path = quote(str(video))
                    hit["thumbnail_url"] = f"/media/thumbnail?path={safe_path}&time={ts}"
                    hit["playback_url"] = f"/media?path={safe_path}#t={max(0, ts - 3)}"

            duration = time.perf_counter() - start_time_search
            result["stats"] = {
                "duration_seconds": round(duration, 3),
                **result.get("modality_breakdown", {}),
            }

            logger.info(
                f"[MultimodalSearch] Returned {len(result.get('results', []))} "
                f"results in {duration:.3f}s"
            )
            return result
        else:
            raise ImportError("SearchAgent undefined")

    except Exception as e:
        logger.error(f"[MultimodalSearch] Error: {e}")
        # Fallback to hybrid search
        results = _normalize_results(
            pipeline.db.search_frames_hybrid(query=q, limit=limit)
        )
        return {
            "query": q,
            "results": results,
            "stats": {"total": len(results), "fallback": True},
            "error": str(e),
            "search_type": "fallback",
        }


# === HITL FEEDBACK ENDPOINT ===

from pydantic import BaseModel


class SearchFeedback(BaseModel):
    """User feedback on search result quality."""
    query: str
    result_id: str
    video_path: str
    timestamp: float
    is_relevant: bool
    feedback_type: str = "binary"  # "binary", "rating", "correction"
    rating: int | None = None  # 1-5 for rating type
    correction: str | None = None  # User-provided correct answer
    notes: str | None = None


@router.post("/search/feedback")
async def submit_search_feedback(
    feedback: SearchFeedback,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> dict:
    """Submit user feedback on a search result (HITL loop).

    This endpoint collects user feedback to:
    1. Identify false positives/negatives
    2. Build training data for reranker improvement
    3. Track search accuracy over time

    Args:
        feedback: The feedback data.
        pipeline: Ingestion pipeline instance.

    Returns:
        Confirmation of feedback submission.
    """
    logger.info(
        f"[HITL] Feedback received: query='{feedback.query[:50]}...' "
        f"is_relevant={feedback.is_relevant} type={feedback.feedback_type}"
    )

    try:
        # Store feedback in Qdrant for later analysis
        import uuid
        from datetime import datetime

        feedback_id = str(uuid.uuid4())
        feedback_data = {
            "feedback_id": feedback_id,
            "query": feedback.query,
            "result_id": feedback.result_id,
            "video_path": feedback.video_path,
            "timestamp": feedback.timestamp,
            "is_relevant": feedback.is_relevant,
            "feedback_type": feedback.feedback_type,
            "rating": feedback.rating,
            "correction": feedback.correction,
            "notes": feedback.notes,
            "submitted_at": datetime.now().isoformat(),
        }

        # Store in a simple JSON file for now (could be moved to Qdrant collection)
        import json
        from pathlib import Path

        feedback_dir = Path("logs/search_feedback")
        feedback_dir.mkdir(parents=True, exist_ok=True)
        feedback_file = feedback_dir / f"{feedback_id}.json"

        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)

        # If this is a correction, we could use it to re-train or boost future results
        if feedback.correction:
            logger.info(f"[HITL] Correction provided: '{feedback.correction}'")

        return {
            "status": "submitted",
            "feedback_id": feedback_id,
            "message": "Thank you for your feedback!",
        }

    except Exception as e:
        logger.error(f"[HITL] Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/search/feedback/stats")
async def get_feedback_stats() -> dict:
    """Get aggregated statistics on search feedback.

    Returns:
        Summary of feedback received.
    """
    import json
    from pathlib import Path

    feedback_dir = Path("logs/search_feedback")
    if not feedback_dir.exists():
        return {"total": 0, "relevant": 0, "irrelevant": 0, "accuracy": None}

    feedback_files = list(feedback_dir.glob("*.json"))
    total = len(feedback_files)
    relevant = 0
    irrelevant = 0

    for f in feedback_files:
        try:
            with open(f) as file:
                data = json.load(file)
                if data.get("is_relevant"):
                    relevant += 1
                else:
                    irrelevant += 1
        except Exception:
            pass

    accuracy = (relevant / total * 100) if total > 0 else None

    return {
        "total": total,
        "relevant": relevant,
        "irrelevant": irrelevant,
        "accuracy_percentage": round(accuracy, 2) if accuracy else None,
    }
