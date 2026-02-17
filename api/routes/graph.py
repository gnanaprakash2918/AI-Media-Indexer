
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.utils.logger import log

router = APIRouter()


class CoOccurrence(BaseModel):
    name: str
    cluster_id: int
    count: int
    relationship_strength: float


class SocialGraphResponse(BaseModel):
    center_person: str
    center_cluster_id: int
    connections: list[CoOccurrence]


@router.get("/social", response_model=SocialGraphResponse)
async def get_social_graph(
    request: Request,
    name: str | None = None,
    cluster_id: int | None = None,
    min_strength: int = 2,
):
    """Get the social graph (co-occurrences) for a specific person.

    Args:
        request: FastAPI request object (to access app state).
        name: Name of the person to look up.
        cluster_id: Direct cluster ID lookup (faster).
        min_strength: Minimum number of co-occurrences to be included.
    """
    pipeline = request.app.state.pipeline
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not name and cluster_id is None:
        raise HTTPException(
            status_code=400, detail="Must provide either name or cluster_id"
        )

    # Resolve Name -> Cluster ID if needed
    if cluster_id is None and name:
        cluster_id = pipeline.db.get_cluster_id_by_name(name)
        if cluster_id is None:
            # Try fuzzy lookup
            cluster_id = pipeline.db.fuzzy_get_cluster_id_by_name(name)

    if cluster_id is None:
        raise HTTPException(
            status_code=404, detail=f"Person '{name}' not found"
        )

    # Get the official name for the response
    resolved_name = (
        pipeline.db.get_face_name_by_cluster(cluster_id)
        or f"Person {cluster_id}"
    )

    try:
        # Fetch co-occurrences from GraphRAG (DB)
        connections_map = pipeline.db.get_person_co_occurrences(cluster_id)

        response_connections = []
        for cid, count in connections_map.items():
            if count < min_strength:
                continue

            c_name = (
                pipeline.db.get_face_name_by_cluster(cid) or f"Person {cid}"
            )

            # Simple normalization for "strength" (0.0 - 1.0) could be added here
            # For now, raw count is the strength

            response_connections.append(
                CoOccurrence(
                    name=c_name,
                    cluster_id=cid,
                    count=count,
                    relationship_strength=float(count),
                )
            )

        # Sort by strength
        response_connections.sort(key=lambda x: x.count, reverse=True)

        return SocialGraphResponse(
            center_person=resolved_name,
            center_cluster_id=cluster_id,
            connections=response_connections,
        )

    except Exception as e:
        log(f"Social graph lookup failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/timeline/{video_path:path}")
async def get_scene_timeline(
    video_path: str,
    request: Request,
):
    """Get the complete scene timeline for a video with transitions.

    Returns ordered scenes with their temporal relationships for
    narrative visualization and GraphRAG queries.

    Args:
        video_path: Path to the video file.
        request: FastAPI request object.

    Returns:
        List of scenes with transitions and people/actions.
    """
    from core.storage.identity_graph import identity_graph

    try:
        timeline = identity_graph.get_scene_timeline(video_path)
        stats = identity_graph.get_graphrag_stats()

        return {
            "video_path": video_path,
            "timeline": timeline,
            "scene_count": len(timeline),
            "graphrag_stats": stats,
        }
    except Exception as e:
        log(f"Scene timeline failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


class SequenceSearchRequest(BaseModel):
    """Request body for temporal sequence search."""

    pattern: list[str]  # e.g., ["speaks", "speaks", "exits"]
    video_path: str | None = None
    person_name: str | None = None


@router.post("/sequence")
async def find_temporal_sequence(
    request: Request,
    body: SequenceSearchRequest,
):
    """Find sequences of events matching a temporal pattern.

    This is the KEY GraphRAG query for temporal sequences like:
    "A speaks" → "B responds" → "A walks away"

    Args:
        body: Search parameters including event pattern.

    Returns:
        List of matching event sequences with timestamps.
    """
    from core.storage.identity_graph import identity_graph

    if not body.pattern:
        raise HTTPException(status_code=400, detail="Pattern cannot be empty")

    try:
        sequences = identity_graph.find_event_sequence(
            pattern=body.pattern,
            media_id=body.video_path,
            identity_name=body.person_name,
        )

        return {
            "pattern": body.pattern,
            "video_filter": body.video_path,
            "person_filter": body.person_name,
            "matches": len(sequences),
            "sequences": sequences,
        }
    except Exception as e:
        log(f"Sequence search failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/stats")
async def get_graphrag_stats(request: Request):
    """Get GraphRAG statistics including scenes, transitions, and events."""
    from core.storage.identity_graph import identity_graph

    try:
        stats = identity_graph.get_graphrag_stats()
        return {
            "status": "ok",
            "stats": stats,
        }
    except Exception as e:
        log(f"GraphRAG stats failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
