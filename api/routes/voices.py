"""API routes for voice/speaker HITL operations."""

from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.deps import get_pipeline
from api.schemas import MergeClustersRequest
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import logger

router = APIRouter()


@router.get("/voices")
async def get_voice_segments(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    media_path: Annotated[
        str | None, Query(description="Filter by media")
    ] = None,
    limit: int = 100,
):
    """Get all voice segments.

    Args:
        media_path: Optional filter for specific video.
        limit: Maximum segments to return.
        pipeline: Ingestion pipeline instance.

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

    Args:
        pipeline: Ingestion pipeline instance.

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

            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "speaker_name": name,  # Frontend expects speaker_name
                    "segment_count": len(segments),
                    "representative": segments[0] if segments else None,
                    "segments": segments,  # All segments, not just 5
                }
            )

        # Sort by: Named first (not None and not "Uncategorized"), then Segment Count
        clusters.sort(
            key=lambda c: (
                c["speaker_name"] is None,  # False (0) comes first
                -c["segment_count"],
            )
        )

        return {
            "clusters": clusters,
            "total_clusters": len(clusters),
        }
    except Exception as e:
        logger.error(f"[Voices] Get clusters failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/voices/merge")
async def merge_voice_clusters(
    request: MergeClustersRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Merge two voice clusters into one.

    Args:
        request: Merge parameters.
        pipeline: Ingestion pipeline instance.

    Returns:
        Merge status and count.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    count = pipeline.db.merge_voice_clusters(
        request.source_cluster_id, request.target_cluster_id
    )
    if count == 0:
        raise HTTPException(
            status_code=404, detail="No segments found to merge"
        )

    return {
        "status": "merged",
        "source_cluster_id": request.source_cluster_id,
        "target_cluster_id": request.target_cluster_id,
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
        pipeline: Ingestion pipeline instance.

    Returns:
        Deletion confirmation.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        from qdrant_client import models

        pipeline.db.client.delete(
            collection_name=pipeline.db.VOICE_COLLECTION,
            points_selector=models.PointIdsList(
                points=cast(list[Any], [segment_id])
            ),
        )
        return {"status": "deleted", "segment_id": segment_id}
    except Exception as e:
        logger.error(f"[Voices] Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/voices/cluster/{cluster_id}")
async def delete_voice_cluster(
    cluster_id: int,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Delete an entire voice cluster and all its segments.

    Args:
        cluster_id: The voice cluster ID to delete.
        pipeline: Ingestion pipeline instance.

    Returns:
        Deletion confirmation with count.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        count = pipeline.db.delete_voice_cluster(cluster_id)
        if count == 0:
            raise HTTPException(
                status_code=404, detail="Cluster not found or empty"
            )
        return {
            "status": "deleted",
            "cluster_id": cluster_id,
            "segments_deleted": count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Voices] Delete cluster failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/voices/cluster")
async def trigger_voice_clustering(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Trigger re-clustering of all voice segments.

    Args:
        pipeline: Ingestion pipeline instance.

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
        return {
            "status": "not_available",
            "message": "Voice clustering module not found",
        }
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
        pipeline: Ingestion pipeline instance.

    Returns:
        New cluster ID.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Use proper atomic cluster ID generation
        new_cluster_id = pipeline.db.get_next_voice_cluster_id()

        from qdrant_client import models

        pipeline.db.client.set_payload(
            collection_name=pipeline.db.VOICE_COLLECTION,
            payload={"voice_cluster_id": new_cluster_id},
            points=models.PointIdsList(points=cast(list[Any], segment_ids)),
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
    cluster_id: Annotated[int, Query(..., description="Target cluster ID")],  # noqa: B008
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Move a voice segment to a different cluster.

    Args:
        segment_id: The segment to move.
        cluster_id: Target cluster ID.
        pipeline: Ingestion pipeline instance.

    Returns:
        Move confirmation.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        from qdrant_client import models

        pipeline.db.client.set_payload(
            collection_name=pipeline.db.VOICE_COLLECTION,
            payload={"voice_cluster_id": cluster_id},
            points=models.PointIdsList(points=cast(list[Any], [segment_id])),
        )
        return {
            "status": "moved",
            "segment_id": segment_id,
            "cluster_id": cluster_id,
        }
    except Exception as e:
        logger.error(f"[Voices] Move failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class ClusterNameRequest(BaseModel):
    """Request schema for naming a voice cluster."""

    name: str


@router.post("/voices/cluster/{cluster_id}/name")
async def name_voice_cluster(
    cluster_id: int,
    request: ClusterNameRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Name a voice cluster and propagate to search.

    Args:
        cluster_id: The ID of the cluster to name.
        request: JSON body containing {"name": "New Name"}.
        pipeline: Ingestion pipeline instance.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        from qdrant_client import models

        # 0. Check for existing cluster with same name (Auto-Merge)
        # Check both 'name' and 'speaker_name' just in case
        existing_resp = pipeline.db.client.scroll(
            collection_name=pipeline.db.VOICE_COLLECTION,
            scroll_filter=models.Filter(
                should=[
                    models.FieldCondition(
                        key="speaker_name",
                        match=models.MatchValue(value=request.name),
                    ),
                    models.FieldCondition(
                        key="name",
                        match=models.MatchValue(value=request.name),
                    ),
                ],
                must_not=[
                    models.FieldCondition(
                        key="voice_cluster_id",
                        match=models.MatchValue(value=cluster_id),
                    )
                ],
            ),
            limit=1,
            with_payload=["voice_cluster_id"],
        )

        points = existing_resp[0]
        if points:
            target_cluster_id = points[0].payload.get("voice_cluster_id")
            if target_cluster_id is not None:
                logger.info(
                    f"Auto-merging voice cluster {cluster_id} into {target_cluster_id} (Name: {request.name})"
                )
                count = pipeline.db.merge_voice_clusters(
                    cluster_id, target_cluster_id
                )
                return {
                    "status": "merged",
                    "source_cluster_id": cluster_id,
                    "target_cluster_id": target_cluster_id,
                    "name": request.name,
                    "segments_moved": count,
                }

        # 1. Update Voice Segments Payload
        # Set both 'speaker_name' and 'name' for consistency
        pipeline.db.client.set_payload(
            collection_name=pipeline.db.VOICE_COLLECTION,
            payload={
                "speaker_name": request.name,
                "name": request.name,
            },  # Fix for get_unresolved_voices using 'name'
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="voice_cluster_id",
                        match=models.MatchValue(value=cluster_id),
                    )
                ]
            ),
        )

        # 2. Propagate to Frames (Search Index)
        count = pipeline.db.re_embed_voice_cluster_frames(
            cluster_id, request.name
        )

        # 3. Identity Linking
        try:
            from core.storage.identity_graph import identity_graph

            identity = identity_graph.get_or_create_identity_by_name(
                request.name
            )
            # Use db method if it exists, but we successfully did it above manually
            # pipeline.db.set_speaker_name(cluster_id, request.name)
        except Exception as e:
            logger.error(f"Identity linking failed: {e}")

        return {
            "status": "updated",
            "cluster_id": cluster_id,
            "name": request.name,
            "frames_updated": count,
        }
    except Exception as e:
        logger.error(f"[Voices] App naming failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/voices/cluster/{cluster_id}/main")
async def mark_main_speaker(
    cluster_id: str,
    segment_id: str,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    is_main: bool = True,
):
    """Mark a voice segment as the representative for its cluster."""
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline invalid")

    try:
        cluster_int = int(cluster_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Cluster ID must be an integer"
        ) from None

    success = pipeline.db.set_speaker_main(cluster_int, segment_id, is_main)
    if not success:
        raise HTTPException(
            status_code=404, detail="Cluster or segment not found"
        )

    return {
        "status": "updated",
        "cluster_id": cluster_id,
        "segment_id": segment_id,
        "is_main_character": is_main,
    }
