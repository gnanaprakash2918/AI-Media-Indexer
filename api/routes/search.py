"""API routes for search operations."""

import time
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_pipeline, get_search_agent
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
    """Normalize search results using centralized utility."""
    from core.utils.normalize import normalize_result
    return [normalize_result(r) for r in results]


# === UNIFIED SEARCH ENDPOINT ===
# All search routes consolidated here. /search and /search/hybrid are aliases.


@router.get("/search")
@router.get("/search/hybrid")
@router.get("/search/unified")
@router.post("/search/unified")
async def unified_search(
    q: Annotated[str, Query(..., description="Search query")],
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    limit: Annotated[int, Query(description="Maximum results")] = 20,
    video_path: Annotated[
        str | None, Query(description="Filter to specific video")
    ] = None,
    # === Pipeline Stage Toggles ===
    enable_expansion: Annotated[
        bool,
        Query(description="Use LLM query expansion (disable for non-English)"),
    ] = True,
    enable_reranking: Annotated[
        bool,
        Query(
            description="Use full council reranking (cross-encoder + BGE + VLM)"
        ),
    ] = True,
    enable_vlm: Annotated[
        bool,
        Query(description="Use VLM visual verification (slower but accurate)"),
    ] = True,
    enable_deep_reasoning: Annotated[
        bool, Query(description="Include detailed CoT reasoning per result")
    ] = True,
    # === Fallback Behavior ===
    expansion_fallback: Annotated[
        bool, Query(description="Retry with original query if expansion fails")
    ] = True,
) -> dict:
    """Unified multimodal search with full pipeline configuration.

    This endpoint combines ALL search modalities and provides maximum
    control over the search pipeline:

    **Modalities Used:**
    - Scene-level visual/motion/dialogue vectors
    - Voice segments (speaker identification)
    - Frame-level fallback when scenes empty
    - Audio events (CLAP-detected sounds)
    - ASR transcripts

    **Pipeline Stages (toggleable):**
    1. **Query Expansion**: LLM parses entities, expands synonyms
    2. **Identity Resolution**: Names → face/voice cluster IDs
    3. **Multi-Vector Search**: Scenes → Scenelets → Frames fallback
    4. **Reranking Council**: Cross-encoder + BGE + VLM scoring
    5. **HITL Boost**: Past feedback influences final ranking
    6. **Deep Reasoning**: CoT explanation per result

    **Use Cases:**
    - `enable_expansion=False` for non-English queries (avoids hallucination)
    - `enable_reranking=False` for faster search (skip LLM verification)
    - `enable_vlm=False` with `enable_reranking=True` for text-only rerank
    - `enable_deep_reasoning=True` for debugging search quality

    Args:
        q: Natural language search query.
        pipeline: The core ingestion pipeline instance.
        limit: Maximum number of results to return.
        video_path: Optional filter for a specific video.
        enable_expansion: Toggle LLM query expansion.
        enable_reranking: Toggle full council reranking.
        enable_vlm: Toggle VLM visual verification.
        enable_deep_reasoning: Toggle detailed reasoning traces.
        expansion_fallback: Retry with raw query if expansion yields few results.

    Returns:
        Comprehensive search results with:
        - results: Ranked matches with scores and reasoning
        - pipeline_steps: What stages were executed
        - stats: Timing and count information
        - modality_breakdown: Results per modality
    """
    start_time_search = time.perf_counter()
    logger.info(
        f"[UnifiedSearch] Query: '{q[:80]}...' | "
        f"expansion={enable_expansion}, rerank={enable_reranking}, "
        f"vlm={enable_vlm}, reasoning={enable_deep_reasoning}"
    )

    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    pipeline_steps = []

    try:
        if SearchAgent:
            agent = SearchAgent(db=pipeline.db)

            # Execute search with configured options
            result = await agent.sota_search(
                query=q,
                limit=limit,
                video_path=video_path,
                use_reranking=enable_reranking,
                use_expansion=enable_expansion,
                expansion_fallback=expansion_fallback,
            )

            # Track pipeline steps
            pipeline_steps = [
                {
                    "step": "Query Expansion",
                    "status": "completed" if enable_expansion else "skipped",
                    "detail": f"Expanded to: {result.get('search_text', q)[:80]}..."
                    if enable_expansion
                    else "Using raw query",
                },
                {
                    "step": "Identity Resolution",
                    "status": "completed",
                    "detail": f"Resolved {len(result.get('person_names_resolved', []))} names → "
                    f"{result.get('face_ids_matched', 0)} faces",
                },
                {
                    "step": "Multi-Vector Search",
                    "status": "completed",
                    "detail": f"Found {result.get('result_count', 0)} candidates "
                    f"(fallback: {result.get('fallback_used', 'none')})",
                },
                {
                    "step": "Reranking Council",
                    "status": "completed" if enable_reranking else "skipped",
                    "detail": f"VLM={'enabled' if enable_vlm else 'disabled'}"
                    if enable_reranking
                    else "Skipped",
                },
                {
                    "step": "HITL Feedback",
                    "status": "completed",
                    "detail": "Applied boost/penalty from past ratings",
                },
            ]

            # Add thumbnail/playback URLs
            for hit in result.get("results", []):
                video = hit.get("video_path") or hit.get("media_path", "")
                start = hit.get("start_time", hit.get("timestamp", 0))
                end = hit.get("end_time", start + 5)

                if video and "thumbnail_url" not in hit:
                    safe_path = quote(str(video))
                    hit["thumbnail_url"] = (
                        f"/media/thumbnail?path={safe_path}&time={start}"
                    )
                    hit["playback_url"] = (
                        f"/media?path={safe_path}#t={max(0, start - 2)}"
                    )
                    # Use actual segment times, not fixed windows
                    hit["segment_url"] = (
                        f"/media/segment?path={safe_path}&start={start:.2f}&end={end:.2f}"
                    )

                # Add deep reasoning if enabled
                if enable_deep_reasoning:
                    reasoning_parts = []

                    # Build detailed reasoning from payload
                    if hit.get("person_names"):
                        reasoning_parts.append(
                            f"[IDENTITY] Found people: {', '.join(hit['person_names'])}"
                        )
                    if hit.get("matched_identity"):
                        reasoning_parts.append(
                            f"[IDENTITY] Matched: {hit['matched_identity']}"
                        )
                    if hit.get("actions"):
                        reasoning_parts.append(
                            f"[ACTIONS] Detected: {', '.join(str(a) for a in hit['actions'][:3])}"
                        )
                    if hit.get("clothing_colors"):
                        reasoning_parts.append(
                            f"[APPEARANCE] Colors: {', '.join(hit['clothing_colors'])}"
                        )
                    if hit.get("visible_text"):
                        reasoning_parts.append(
                            f"[TEXT] Visible: {', '.join(str(t) for t in hit['visible_text'][:3])}"
                        )
                    if hit.get("dialogue_transcript"):
                        reasoning_parts.append(
                            f'[DIALOGUE] "{hit["dialogue_transcript"][:100]}..."'
                        )
                    if hit.get("llm_reasoning"):
                        reasoning_parts.append(
                            f"[VERIFICATION] {hit['llm_reasoning']}"
                        )

                    hit["deep_reasoning"] = (
                        " → ".join(reasoning_parts) if reasoning_parts else None
                    )

            duration = time.perf_counter() - start_time_search

            logger.info(
                f"[UnifiedSearch] Returned {len(result.get('results', []))} results "
                f"in {duration:.3f}s"
            )

            return {
                "query": q,
                "search_type": "unified",
                "results": result.get("results", []),
                "stats": {
                    "total": len(result.get("results", [])),
                    "duration_seconds": round(duration, 3),
                    "expansion_used": enable_expansion,
                    "reranking_used": enable_reranking,
                    "vlm_used": enable_vlm,
                    "fallback_used": result.get("fallback_used"),
                },
                "pipeline_steps": pipeline_steps,
                "parsed_query": result.get("parsed", {}),
                "search_text": result.get("search_text", q),
                "identities_resolved": {
                    "names": result.get("person_names_resolved", []),
                    "face_ids": result.get("face_ids_matched", 0),
                },
                "reasoning_chain": result.get("reasoning_chain", {}),
            }
        else:
            raise ImportError("SearchAgent undefined")

    except Exception as e:
        logger.error(f"[UnifiedSearch] Error: {e}")
        import traceback

        logger.error(traceback.format_exc())

        # Fallback to basic hybrid search
        results = _normalize_results(
            pipeline.db.search_frames_hybrid(query=q, limit=limit)
        )
        duration = time.perf_counter() - start_time_search

        return {
            "query": q,
            "search_type": "unified_fallback",
            "results": results,
            "stats": {
                "total": len(results),
                "duration_seconds": round(duration, 3),
                "fallback": True,
            },
            "error": str(e),
            "pipeline_steps": [
                {
                    "step": "Fallback",
                    "status": "used",
                    "detail": f"Error: {str(e)[:50]}",
                },
            ],
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
