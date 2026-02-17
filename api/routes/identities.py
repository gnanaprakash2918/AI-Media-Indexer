"""API routes for identity management and cluster operations."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_pipeline
from api.schemas import (
    BulkApproveRequest,
    CreateClusterRequest,
    IdentityMergeRequest,
    IdentityRenameRequest,
    MergeClustersRequest,
    MoveFacesRequest,
    NameFaceRequest,
)
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import logger

router = APIRouter()


@router.get("/identities")
async def list_identities() -> dict:
    """Retrieves all recognized identities and their face track counts.

    Returns:
        A dictionary containing a list of identity records and the total count.
    """
    from core.storage.identity_graph import identity_graph

    identities = identity_graph.get_all_identities()
    result = [
        {
            "id": ident.id,
            "name": ident.name,
            "is_verified": ident.is_verified,
            "face_track_count": getattr(ident, "face_track_count", 0),
            "voice_track_count": getattr(ident, "voice_track_count", 0),
            "created_at": ident.created_at,
        }
        for ident in identities
    ]
    return {"identities": result, "total": len(result)}


@router.get("/identities/names")
async def list_all_names(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Get all HITL-assigned names (faces and speakers).

    Returns:
        List of unique names.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        names = pipeline.db.get_all_hitl_names()
        return {"names": sorted(names)}
    except Exception as e:
        logger.error(f"[Identities] List names failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/identities/suggestions")
async def suggest_merges(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    limit_frames: int = 5000,
    music_percentage: float = 0.0,
) -> dict:
    """Generates identity merge suggestions (HITL).

    Aggregates data from face clusters, voice clusters, and NER entities
    to suggest:
    1. Face-Voice merges (Temporal Co-occurrence)
    2. Face-Face merges (Name Similarity)
    3. Face-NER matches (Entity Co-occurrence) - NEW!

    Args:
        pipeline: Ingestion pipeline instance.
        limit_frames: Number of recent frames to analyze for co-occurrence.
        music_percentage: Audio music percentage (0-100) for the video.
            When high (>50%), face-voice link confidence is dampened to
            prevent lip-sync actors being linked to playback singers.

    Returns:
        List of suggestion objects with confidence and strict_mode flags.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline invalid")

    try:
        from core.processing.identity_linker import get_identity_linker

        # 1. Gather Data
        # Face Clusters (Unresolved + Named)
        # face_clusters = pipeline.db.get_unresolved_faces(limit=1000)
        # Also grab named ones if needed, but usually we merge unnamed -> named.
        # For simple version, let's stick to what get_unresolved_faces returns (which has times).
        # But wait, get_unresolved_faces returns list of faces, we need clusters.
        # Let's use get_face_clusters logic or fetch aggregated.
        # Ideally we want aggregated cluster info (times, name if exists).

        # Re-using logic from faces.py get_face_clusters roughly:
        faces_raw = pipeline.db.get_unresolved_faces(limit=2000)
        # Group by cluster
        clusters_map = {}
        for f in faces_raw:
            cid = f.get("cluster_id")
            if cid not in clusters_map:
                clusters_map[cid] = {
                    "cluster_id": cid,
                    "timestamps": [],
                    "name": f.get("name"),
                }
            ts = f.get("timestamp")
            if ts:
                clusters_map[cid]["timestamps"].append(ts)

        face_cluster_list = list(clusters_map.values())

        # Voice Clusters
        voice_segments = pipeline.db.get_all_voice_segments(limit=2000)
        voice_map = {}
        for v in voice_segments:
            cid = v.get("voice_cluster_id")
            if cid:
                if cid not in voice_map:
                    voice_map[cid] = {
                        "cluster_id": cid,
                        "timestamps": [],
                        "name": v.get("speaker_name"),
                    }
                voice_map[cid]["timestamps"].append(
                    {"start": v.get("start"), "end": v.get("end")}
                )
        voice_cluster_list = list(voice_map.values())

        # NER Co-occurrences
        entity_co = pipeline.db.get_entity_co_occurrences(
            limit_frames=limit_frames
        )

        # 2. Link
        linker = get_identity_linker()
        suggestions = linker.get_all_suggestions(
            face_clusters=face_cluster_list,
            voice_clusters=voice_cluster_list,
            entity_occurrences=entity_co,
            music_percentage=music_percentage,
        )

        return {"suggestions": suggestions, "count": len(suggestions)}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/identities/{identity_id}/merge")
async def merge_identities(identity_id: str, req: IdentityMergeRequest) -> dict:
    """Merges a source identity into a target identity cluster.

    Args:
        identity_id: The ID of the source identity to merge.
        req: The request payload specifying the target identity.

    Returns:
        A dictionary confirming the merge operation.

    Raises:
        HTTPException: If the merge operation fails.
    """
    from core.storage.identity_graph import identity_graph

    try:
        identity_graph.merge_identities(identity_id, req.target_identity_id)
        return {
            "status": "merged",
            "source": identity_id,
            "target": req.target_identity_id,
        }
    except Exception as e:
        logger.error(f"[Identities] Merge failed: {e}")
        raise HTTPException(status_code=400, detail="Identity merge failed") from e


@router.patch("/identities/{identity_id}")
async def rename_identity(
    identity_id: str,
    req: IdentityRenameRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> dict:
    """Updates the display name of a recognized identity.

    Args:
        identity_id: The ID of the identity to rename.
        req: The request payload containing the new name.

    Returns:
        A dictionary confirming the name update.

    Raises:
        HTTPException: If the identity is not found.
    """
    from core.storage.identity_graph import identity_graph

    identity = identity_graph.get_identity(identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found")
    identity_graph.update_identity_name(identity_id, req.name)

    # Also update SAM 3 Masklets (The "Track Everywhere" promise)
    try:
        if pipeline and pipeline.db:
             # We need the OLD name to find masklets. identity object has it.
             old_name = identity.name
             count = pipeline.db.update_masklet_concept(old_name, req.name)
             if count > 0:
                 logger.info(f"[Identity] Also renamed {count} masklets for {req.name}")
    except Exception as e:
        logger.warning(f"[Identity] Failed to propagate rename to masklets: {e}")

    return {"status": "renamed", "id": identity_id, "name": req.name}


@router.delete("/identities/{identity_id}")
async def delete_identity(
    identity_id: str,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> dict:
    """Permanently deletes an identity record and cleans up references."""
    from core.storage.identity_graph import identity_graph

    identity = identity_graph.get_identity(identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found")

    # Clean up face/voice references in Qdrant before deleting from graph
    if identity.name and pipeline and pipeline.db:
        try:
            from qdrant_client import models

            # Clear name from face points
            pipeline.db.client.set_payload(
                collection_name=pipeline.db.FACES_COLLECTION,
                payload={"name": None},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="name",
                            match=models.MatchValue(value=identity.name),
                        )
                    ]
                ),
            )
            # Clear speaker_name from voice points
            pipeline.db.client.set_payload(
                collection_name=pipeline.db.VOICE_COLLECTION,
                payload={"speaker_name": None, "name": None},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="speaker_name",
                            match=models.MatchValue(value=identity.name),
                        )
                    ]
                ),
            )
        except Exception as e:
            logger.warning(
                f"[Identity] Failed to clean up Qdrant refs for {identity.name}: {e}"
            )

    identity_graph.delete_identity(identity_id)
    return {"status": "deleted", "id": identity_id}


# === CLUSTER OPERATIONS ===
@router.post("/clusters/create")
async def create_cluster(
    req: CreateClusterRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> dict:
    """Creates a new manual identity cluster.

    Args:
        req: The request payload for cluster creation.
        pipeline: The core ingestion pipeline instance.

    Returns:
        A dictionary containing the new cluster ID and status.

    Raises:
        HTTPException: If the database is not ready.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="DB not ready")

    raise HTTPException(
        status_code=501,
        detail="Manual cluster creation is not yet implemented",
    )


@router.post("/faces/move")
async def move_faces(
    req: MoveFacesRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> dict:
    """Moves specific face points to a different identity cluster.

    Args:
        req: The request payload specifying face IDs and target cluster.
        pipeline: The core ingestion pipeline instance.

    Returns:
        A dictionary confirming the move operation and count.

    Raises:
        HTTPException: If the database is not ready or move fails.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="DB not ready")

    try:
        from typing import Any, cast

        # Use Qdrant's set_payload to update cluster_id for specified face IDs
        if req.face_ids:
            # Pylance strict check: Ensure valid list passed to constructor
            points_list = cast(list[Any], req.face_ids)
            pipeline.db.client.set_payload(
                collection_name=pipeline.db.FACES_COLLECTION,
                payload={"cluster_id": req.target_cluster_id},
                points=points_list,
            )
        return {
            "status": "moved",
            "count": len(req.face_ids),
            "target_cluster": req.target_cluster_id,
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to move faces: {e}"
        ) from e


@router.post("/clusters/merge")
async def merge_clusters(
    req: MergeClustersRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> dict:
    """Merges two face clusters at the storage/database level.

    Args:
        req: The request payload specifying source and target clusters.
        pipeline: The core ingestion pipeline instance.

    Returns:
        A dictionary confirming the merge status.

    Raises:
        HTTPException: If the database is not ready or merge fails.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="DB not ready")

    try:
        count = pipeline.db.merge_face_clusters(
            req.source_cluster_id, req.target_cluster_id
        )
        return {
            "status": "merged",
            "source_cluster": req.source_cluster_id,
            "target_cluster": req.target_cluster_id,
            "faces_moved": count,
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to merge clusters: {e}"
        ) from e


@router.post("/clusters/approve")
async def bulk_approve(
    req: BulkApproveRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> dict:
    """Batch approves recognized face clusters for production indexing.

    Args:
        req: The request payload containing cluster IDs to approve.
        pipeline: The core ingestion pipeline instance.

    Returns:
        A dictionary confirming the approval status and count.
    """
    # Logic for approval
    raise HTTPException(
        status_code=501,
        detail="Bulk approval is not yet implemented",
    )




@router.post("/faces/cluster/{cluster_id}/main")
async def mark_main_character(
    cluster_id: int,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    is_main: bool = True,
):
    """Mark a face cluster as main character for prioritized ranking.

    Args:
        cluster_id: The face cluster ID to mark.
        pipeline: The core ingestion pipeline instance.
        is_main: Whether this is a main character (default True).

    Returns:
        A dictionary confirming the update.

    Raises:
        HTTPException: If the cluster is not found or update fails.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline invalid")

    success = pipeline.db.set_face_main(cluster_id, is_main)
    if not success:
        raise HTTPException(status_code=404, detail="Cluster not found")

    return {
        "status": "updated",
        "cluster_id": cluster_id,
        "is_main_character": is_main,
    }
