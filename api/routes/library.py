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
        # Scroll with minimal payload to build video index
        videos: dict[str, dict] = {}
        offset = None
        while True:
            resp = pipeline.db.client.scroll(
                collection_name=pipeline.db.MEDIA_COLLECTION,
                scroll_filter=None,
                limit=1000,
                offset=offset,
                with_payload=["video_path", "timestamp"],
                with_vectors=False,
            )
            points, next_offset = resp
            for point in points:
                payload = point.payload or {}
                path = payload.get("video_path")
                if not path:
                    continue
                if path not in videos:
                    videos[path] = {
                        "path": path,
                        "video_path": path,
                        "frame_count": 0,
                        "first_timestamp": payload.get("timestamp", 0),
                    }
                videos[path]["frame_count"] += 1

            if next_offset is None:
                break
            offset = next_offset

        return {
            "media": list(videos.values()),
            "total": len(videos),
        }
    except Exception as e:
        logger.error(f"[Library] Get library failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


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
        errors = {}

        # Collections to clean up and their path field names
        collections = [
            (pipeline.db.MEDIA_COLLECTION, "video_path", "frames"),
            (pipeline.db.VOICE_COLLECTION, "media_path", "voices"),
            (pipeline.db.FACES_COLLECTION, "media_path", "faces"),
        ]

        for collection_name, path_field, label in collections:
            try:
                pipeline.db.client.delete(
                    collection_name=collection_name,
                    points_selector=qmodels.FilterSelector(
                        filter=qmodels.Filter(
                            must=[
                                qmodels.FieldCondition(
                                    key=path_field,
                                    match=qmodels.MatchValue(value=path),
                                )
                            ]
                        )
                    ),
                )
                deleted_counts[label] = "deleted"
            except Exception as e:
                logger.error(f"[Library] Failed to delete {label} for {path}: {e}")
                errors[label] = str(e)

        status = "deleted" if not errors else "partial_failure"
        result = {
            "status": status,
            "path": path,
            "collections_cleaned": deleted_counts,
        }
        if errors:
            result["errors"] = errors

        return result
    except Exception as e:
        logger.error(f"[Library] Delete failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


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

        # Count unique videos via paginated scroll
        try:
            unique_videos: set[str] = set()
            offset = None
            while True:
                resp = pipeline.db.client.scroll(
                    collection_name=pipeline.db.MEDIA_COLLECTION,
                    limit=1000,
                    offset=offset,
                    with_payload=["video_path"],
                    with_vectors=False,
                )
                points, next_offset = resp
                for p in points:
                    vp = (p.payload or {}).get("video_path")
                    if vp:
                        unique_videos.add(vp)
                if next_offset is None:
                    break
                offset = next_offset
            stats["videos"] = len(unique_videos)
        except Exception:
            stats["videos"] = 0

        return stats
    except Exception as e:
        logger.error(f"[Stats] Get stats failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/config")
async def get_config():
    """Get current configuration.

    Returns:
        Current settings (non-sensitive).
    """
    from config import settings

    return {
        "llm_provider": settings.llm_provider,
        "ollama_model": settings.ollama_text_model,
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
        raise HTTPException(status_code=500, detail="Internal server error") from e
