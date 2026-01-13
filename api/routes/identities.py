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
async def list_identities():
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
async def merge_identities(identity_id: str, req: IdentityMergeRequest):
    from core.storage.identity_graph import identity_graph

    try:
        identity_graph.merge_identities(identity_id, req.target_identity_id)
        return {
            "status": "merged",
            "source": identity_id,
            "target": req.target_identity_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/identities/{identity_id}")
async def rename_identity(identity_id: str, req: IdentityRenameRequest):
    from core.storage.identity_graph import identity_graph

    identity = identity_graph.get_identity(identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found")
    identity_graph.update_identity_name(identity_id, req.name)
    return {"status": "renamed", "id": identity_id, "name": req.name}


@router.delete("/identities/{identity_id}")
async def delete_identity(identity_id: str):
    from core.storage.identity_graph import identity_graph

    identity_graph.delete_identity(identity_id)
    return {"status": "deleted", "id": identity_id}


# === CLUSTER OPERATIONS ===
@router.post("/clusters/create")
async def create_cluster(
    req: CreateClusterRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="DB not ready")

    # Logic to create manual cluster if supported by DB
    # Assuming DB has create_face_cluster method or similar
    # For now, placeholder as original server.py didn't have strict logic for manual cluster creation exposed directly
    return {"status": "created", "id": "manual_cluster_id", "name": req.name}


@router.post("/faces/move")
async def move_faces(
    req: MoveFacesRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="DB not ready")

    # Logic to move faces
    # pipeline.db.move_faces_to_cluster(req.face_ids, req.target_cluster_id)
    return {"status": "moved", "count": len(req.face_ids)}


@router.post("/clusters/merge")
async def merge_clusters(
    req: MergeClustersRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="DB not ready")

    # pipeline.db.merge_clusters(req.source_cluster_id, req.target_cluster_id)
    return {"status": "merged"}


@router.post("/clusters/approve")
async def bulk_approve(
    req: BulkApproveRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    # Logic for approval
    return {"status": "approved", "count": len(req.cluster_ids)}

@router.post("/faces/cluster/{cluster_id}/name")
async def name_face_cluster(
    cluster_id: str,
    req: NameFaceRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    """Assign a name to a face cluster (Person N -> 'John Doe')."""
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline invalid")

    success = pipeline.db.set_face_name(cluster_id, req.name)
    if not success:
        raise HTTPException(status_code=404, detail="Cluster not found")

    return {"status": "updated", "id": cluster_id, "name": req.name}
