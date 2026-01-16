"""API routes for library and system operations."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_pipeline
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import logger

router = APIRouter()


@router.get("/library")
async def get_library(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Get all indexed media in the library.

    Args:
        pipeline: Ingestion pipeline instance.

    Returns:
        List of indexed videos with metadata.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Get unique video paths from frames collection
        resp = pipeline.db.client.scroll(
            collection_name=pipeline.db.MEDIA_COLLECTION,
            scroll_filter=None,
            limit=10000,
            with_payload=["video_path", "action", "timestamp"],
            with_vectors=False,
        )

        # Group by video_path
        videos: dict[str, dict] = {}
        for point in resp[0]:
            payload = point.payload or {}
            path = payload.get("video_path")
            if path and path not in videos:
                videos[path] = {
                    "path": path,
                    "video_path": path,
                    "frame_count": 0,
                    "first_timestamp": payload.get("timestamp", 0),
                }
            if path:
                videos[path]["frame_count"] += 1

        return {
            "media": list(videos.values()),
            "total": len(videos),
        }
    except Exception as e:
        logger.error(f"[Library] Get library failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/library")
async def delete_from_library(
    path: Annotated[str, Query(..., description="Video path to delete")],
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Remove a video and all its data from the library.

    Args:
        path: The video path to remove.
        pipeline: Ingestion pipeline instance.

    Returns:
        Deletion confirmation.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        from qdrant_client.http import models as qmodels

        deleted_counts = {}

        # Delete from frames collection
        try:
            pipeline.db.client.delete(
                collection_name=pipeline.db.MEDIA_COLLECTION,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="video_path",
                                match=qmodels.MatchValue(value=path),
                            )
                        ]
                    )
                ),
            )
            deleted_counts["frames"] = "deleted"
        except Exception:
            pass

        # Delete from voices collection
        try:
            pipeline.db.client.delete(
                collection_name=pipeline.db.VOICE_COLLECTION,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="media_path",
                                match=qmodels.MatchValue(value=path),
                            )
                        ]
                    )
                ),
            )
            deleted_counts["voices"] = "deleted"
        except Exception:
            pass

        # Delete from faces collection
        try:
            pipeline.db.client.delete(
                collection_name=pipeline.db.FACES_COLLECTION,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="media_path",
                                match=qmodels.MatchValue(value=path),
                            )
                        ]
                    )
                ),
            )
            deleted_counts["faces"] = "deleted"
        except Exception:
            pass

        return {
            "status": "deleted",
            "path": path,
            "collections_cleaned": deleted_counts,
        }
    except Exception as e:
        logger.error(f"[Library] Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stats")
async def get_stats(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Get system statistics.

    Args:
        pipeline: Ingestion pipeline instance.

    Returns:
        Counts of indexed items across collections.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        stats = {}

        # Count frames
        try:
            info = pipeline.db.client.get_collection(
                pipeline.db.MEDIA_COLLECTION
            )
            stats["frames"] = info.points_count
        except Exception:
            stats["frames"] = 0

        # Count faces
        try:
            info = pipeline.db.client.get_collection(
                pipeline.db.FACES_COLLECTION
            )
            stats["faces"] = info.points_count
        except Exception:
            stats["faces"] = 0

        # Count voice segments
        try:
            info = pipeline.db.client.get_collection(
                pipeline.db.VOICE_COLLECTION
            )
            stats["voice_segments"] = info.points_count
        except Exception:
            stats["voice_segments"] = 0

        # Count unique videos
        try:
            resp = pipeline.db.client.scroll(
                collection_name=pipeline.db.MEDIA_COLLECTION,
                limit=10000,
                with_payload=["video_path"],
                with_vectors=False,
            )
            unique_videos = set()
            for p in resp[0]:
                vp = (p.payload or {}).get("video_path")
                if vp:
                    unique_videos.add(vp)
            stats["videos"] = len(unique_videos)
        except Exception:
            stats["videos"] = 0

        return stats
    except Exception as e:
        logger.error(f"[Stats] Get stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/config")
async def get_config():
    """Get current configuration.

    Returns:
        Current settings (non-sensitive).
    """
    from config import settings

    return {
        "llm_provider": settings.llm_provider,
        "ollama_model": settings.ollama_model,
        "frame_interval": settings.frame_interval,
        "qdrant_backend": settings.qdrant_backend,
        "embedding_model": settings.embedding_model_override or "auto",
    }


@router.get("/search/by-name")
async def search_by_name(
    name: Annotated[str, Query(..., description="Person name to search")],
    limit: int = 20,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)] = None,
):
    """Search for frames by person name.

    Args:
        name: The person name to search for.
        limit: Maximum results.
        pipeline: Ingestion pipeline instance.

    Returns:
        Frames containing the named person.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Find cluster ID for name
        cluster_id = pipeline.db.get_cluster_id_by_name(name)

        if not cluster_id:
            return {"results": [], "message": f"No person named '{name}' found"}

        # Get frames with this cluster
        frames = pipeline.db.get_frames_by_face_cluster(cluster_id, limit=limit)

        return {
            "name": name,
            "cluster_id": cluster_id,
            "results": frames,
            "count": len(frames),
        }
    except Exception as e:
        logger.error(f"[Search] By-name failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
