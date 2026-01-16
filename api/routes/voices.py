"""API routes for voice/speaker HITL operations."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_pipeline
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import logger

router = APIRouter()


@router.get("/voices")
async def get_voice_segments(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    media_path: Annotated[str | None, Query(description="Filter by media")] = None,
    limit: int = 100,
):
    """Get all voice segments.

    Args:
        media_path: Optional filter for specific video.
        limit: Maximum segments to return.

    Returns:
        List of voice segments with speaker info.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        if media_path:
            segments = pipeline.db.get_voice_segments_for_media(media_path)
        else:
            segments = pipeline.db.get_all_voice_segments(limit=limit)
        return {"segments": segments, "count": len(segments)}
    except Exception as e:
        logger.error(f"[Voices] Get segments failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/voices/clusters")
async def get_voice_clusters(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Get all voice segments grouped by cluster.

    Returns:
        Dictionary with clusters and their segments.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        grouped = pipeline.db.get_voices_grouped_by_cluster(limit=500)

        clusters = []
        for cluster_id, segments in grouped.items():
            # Get name from first segment if available
            name = None
            for seg in segments:
                if seg.get("speaker_name"):
                    name = seg["speaker_name"]
                    break

            clusters.append({
                "cluster_id": cluster_id,
                "speaker_name": name,  # Frontend expects speaker_name
                "segment_count": len(segments),
                "representative": segments[0] if segments else None,
                "segments": segments,  # All segments, not just 5
            })

        # Sort by: named speakers first, then by segment_count descending
        clusters.sort(key=lambda c: (c["speaker_name"] is None, -c["segment_count"]))

        return {
            "clusters": clusters,
            "total_clusters": len(clusters),
        }
    except Exception as e:
        logger.error(f"[Voices] Get clusters failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/voices/merge")
async def merge_voice_clusters(
    source_cluster_id: int,
    target_cluster_id: int,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Merge two voice clusters into one.

    Args:
        source_cluster_id: Cluster to merge FROM.
        target_cluster_id: Cluster to merge INTO.

    Returns:
        Merge status and count.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    count = pipeline.db.merge_voice_clusters(source_cluster_id, target_cluster_id)
    if count == 0:
        raise HTTPException(status_code=404, detail="No segments found to merge")

    return {
        "status": "merged",
        "source_cluster_id": source_cluster_id,
        "target_cluster_id": target_cluster_id,
        "segments_moved": count,
    }


@router.delete("/voices/{segment_id}")
async def delete_voice_segment(
    segment_id: str,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Delete a voice segment.

    Args:
        segment_id: The segment ID to delete.

    Returns:
        Deletion confirmation.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        pipeline.db.client.delete(
            collection_name=pipeline.db.VOICE_COLLECTION,
            points_selector=[segment_id],
        )
        return {"status": "deleted", "segment_id": segment_id}
    except Exception as e:
        logger.error(f"[Voices] Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/voices/cluster")
async def trigger_voice_clustering(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Trigger re-clustering of all voice segments.

    Returns:
        Clustering status.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        from core.processing.voice_clustering import cluster_voices

        stats = await cluster_voices(pipeline.db)
        return {"status": "completed", **stats}
    except ImportError:
        return {"status": "not_available", "message": "Voice clustering module not found"}
    except Exception as e:
        logger.error(f"[Voices] Clustering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/voices/new-cluster")
async def create_new_voice_cluster(
    segment_ids: list[str],
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Create a new cluster from selected voice segments.

    Args:
        segment_ids: List of segment IDs to move.

    Returns:
        New cluster ID.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        import random
        new_cluster_id = random.randint(10000, 99999)

        pipeline.db.client.set_payload(
            collection_name=pipeline.db.VOICE_COLLECTION,
            payload={"voice_cluster_id": new_cluster_id},
            points=segment_ids,
        )

        return {
            "status": "created",
            "new_cluster_id": new_cluster_id,
            "segments_moved": len(segment_ids),
        }
    except Exception as e:
        logger.error(f"[Voices] New cluster failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/voices/{segment_id}/cluster")
async def move_voice_to_cluster(
    segment_id: str,
    cluster_id: Annotated[int, Query(..., description="Target cluster ID")],
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Move a voice segment to a different cluster.

    Args:
        segment_id: The segment to move.
        cluster_id: Target cluster ID.

    Returns:
        Move confirmation.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        pipeline.db.client.set_payload(
            collection_name=pipeline.db.VOICE_COLLECTION,
            payload={"voice_cluster_id": cluster_id},
            points=[segment_id],
        )
        return {"status": "moved", "segment_id": segment_id, "cluster_id": cluster_id}
    except Exception as e:
        logger.error(f"[Voices] Move failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
