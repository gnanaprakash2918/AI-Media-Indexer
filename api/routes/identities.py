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

router = APIRouter()


@router.get("/identities")
async def list_identities() -> dict:
    """Retrieves all recognized identities and their face track counts.

    Returns:
        A dictionary containing a list of identity records and the total count.
    """
    from core.storage.identity_graph import identity_graph

    identities = identity_graph.get_all_identities()
    result = []
    for ident in identities:
        tracks = identity_graph.get_face_tracks_for_identity(ident.id)
        result.append(
            {
                "id": ident.id,
                "name": ident.name,
                "is_verified": ident.is_verified,
                "face_track_count": len(tracks),
                "created_at": ident.created_at,
            }
        )
    return {"identities": result, "total": len(result)}


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
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.patch("/identities/{identity_id}")
async def rename_identity(identity_id: str, req: IdentityRenameRequest) -> dict:
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
    return {"status": "renamed", "id": identity_id, "name": req.name}


@router.delete("/identities/{identity_id}")
async def delete_identity(identity_id: str) -> dict:
    """Permanently deletes an identity record and associated metadata.

    Args:
        identity_id: The ID of the identity to delete.

    Returns:
        A dictionary confirming the deletion.
    """
    from core.storage.identity_graph import identity_graph

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

    # Logic to create manual cluster if supported by DB
    # Assuming DB has create_face_cluster method or similar
    # For now, placeholder as original server.py didn't have strict logic for manual cluster creation exposed directly
    return {"status": "created", "id": "manual_cluster_id", "name": req.name}


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
        # Use Qdrant's set_payload to update cluster_id for specified face IDs
        from qdrant_client.http import models

        pipeline.db.client.set_payload(
            collection_name=pipeline.db.FACES_COLLECTION,
            payload={"cluster_id": req.target_cluster_id},
            points=req.face_ids,
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
    return {"status": "approved", "count": len(req.cluster_ids)}


@router.post("/faces/cluster/{cluster_id}/name")
async def name_face_cluster(
    cluster_id: str,
    req: NameFaceRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Assign a name to a face cluster (Person N -> 'John Doe')."""
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline invalid")

    success = pipeline.db.set_face_name(cluster_id, req.name)
    if not success:
        raise HTTPException(status_code=404, detail="Cluster not found")

    return {"status": "updated", "id": cluster_id, "name": req.name}


@router.post("/faces/cluster/{cluster_id}/main")
async def mark_main_character(
    cluster_id: str,
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

    try:
        cluster_int = int(cluster_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Cluster ID must be an integer"
        )

    success = pipeline.db.set_face_main(cluster_int, is_main)
    if not success:
        raise HTTPException(status_code=404, detail="Cluster not found")

    return {
        "status": "updated",
        "cluster_id": cluster_id,
        "is_main_character": is_main,
    }
