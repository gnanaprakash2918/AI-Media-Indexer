"""API routes for face HITL operations."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_pipeline
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import logger

router = APIRouter()


@router.get("/faces/clusters")
async def get_face_clusters(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Get all face clusters grouped by cluster_id.

    Returns:
        Dictionary with clusters and their representative faces.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Get all faces grouped by cluster
        faces = pipeline.db.get_unresolved_faces(limit=1000)

        # Also get named faces
        named = []
        try:
            resp = pipeline.db.client.scroll(
                collection_name=pipeline.db.FACES_COLLECTION,
                scroll_filter=None,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            for point in resp[0]:
                payload = point.payload or {}
                if payload.get("name"):
                    named.append({
                        "id": str(point.id),
                        "cluster_id": payload.get("cluster_id"),
                        "name": payload.get("name"),
                        "media_path": payload.get("media_path"),
                        "timestamp": payload.get("timestamp"),
                        "thumbnail_path": payload.get("thumbnail_path"),
                        "is_main": payload.get("is_main", False),
                    })
        except Exception:
            pass

        # Group by cluster_id
        clusters: dict[int, list] = {}
        for face in faces + named:
            cid = face.get("cluster_id", -1)
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(face)

        return {
            "clusters": [
                {
                    "cluster_id": cid,
                    "name": faces[0].get("name") if faces else None,
                    "count": len(faces),
                    "is_main": any(f.get("is_main") for f in faces),
                    "faces": faces[:5],  # Representative samples
                }
                for cid, faces in clusters.items()
            ],
            "total_clusters": len(clusters),
        }
    except Exception as e:
        logger.error(f"[Faces] Get clusters failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/faces/unresolved")
async def get_unresolved_faces(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    limit: int = 50,
):
    """Get faces without assigned names for HITL labeling.

    Args:
        limit: Maximum number of unresolved faces to return.

    Returns:
        List of faces needing naming.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    faces = pipeline.db.get_unresolved_faces(limit=limit)
    return {"faces": faces, "count": len(faces)}


@router.get("/faces/named")
async def get_named_faces(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Get all faces with assigned names.

    Returns:
        List of named face clusters.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        from qdrant_client.http import models

        resp = pipeline.db.client.scroll(
            collection_name=pipeline.db.FACES_COLLECTION,
            scroll_filter=models.Filter(
                must_not=[
                    models.IsNullCondition(
                        is_null=models.PayloadField(key="name")
                    )
                ]
            ),
            limit=500,
            with_payload=True,
            with_vectors=False,
        )

        # Group by name
        named: dict[str, list] = {}
        for point in resp[0]:
            payload = point.payload or {}
            name = payload.get("name")
            if name:
                if name not in named:
                    named[name] = []
                named[name].append({
                    "id": str(point.id),
                    "cluster_id": payload.get("cluster_id"),
                    "media_path": payload.get("media_path"),
                    "timestamp": payload.get("timestamp"),
                    "thumbnail_path": payload.get("thumbnail_path"),
                })

        return {
            "named_clusters": [
                {"name": name, "faces": faces, "count": len(faces)}
                for name, faces in named.items()
            ],
            "total_named": len(named),
        }
    except Exception as e:
        logger.error(f"[Faces] Get named failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/faces/merge")
async def merge_face_clusters(
    source_cluster_id: int,
    target_cluster_id: int,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Merge two face clusters into one.

    Args:
        source_cluster_id: Cluster to merge FROM (will be dissolved).
        target_cluster_id: Cluster to merge INTO.

    Returns:
        Merge status and count of moved faces.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    count = pipeline.db.merge_face_clusters(source_cluster_id, target_cluster_id)
    if count == 0:
        raise HTTPException(status_code=404, detail="No faces found to merge")

    return {
        "status": "merged",
        "source_cluster_id": source_cluster_id,
        "target_cluster_id": target_cluster_id,
        "faces_moved": count,
    }


@router.delete("/faces/{face_id}")
async def delete_face(
    face_id: str,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Delete a single face from the database.

    Args:
        face_id: The face point ID to delete.

    Returns:
        Deletion confirmation.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        pipeline.db.client.delete(
            collection_name=pipeline.db.FACES_COLLECTION,
            points_selector=[face_id],
        )
        return {"status": "deleted", "face_id": face_id}
    except Exception as e:
        logger.error(f"[Faces] Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/faces/cluster")
async def trigger_face_clustering(
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Trigger re-clustering of all faces.

    Returns:
        Clustering status.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Run clustering
        from core.processing.clustering import cluster_faces

        stats = await cluster_faces(pipeline.db)
        return {"status": "completed", **stats}
    except ImportError:
        return {"status": "not_available", "message": "Clustering module not found"}
    except Exception as e:
        logger.error(f"[Faces] Clustering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/faces/new-cluster")
async def create_new_face_cluster(
    face_ids: list[str],
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Create a new cluster from selected faces.

    Args:
        face_ids: List of face IDs to move to new cluster.

    Returns:
        New cluster ID.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        import random
        new_cluster_id = random.randint(10000, 99999)

        pipeline.db.client.set_payload(
            collection_name=pipeline.db.FACES_COLLECTION,
            payload={"cluster_id": new_cluster_id},
            points=face_ids,
        )

        return {
            "status": "created",
            "new_cluster_id": new_cluster_id,
            "faces_moved": len(face_ids),
        }
    except Exception as e:
        logger.error(f"[Faces] New cluster failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/faces/{face_id}/cluster")
async def move_face_to_cluster(
    face_id: str,
    cluster_id: Annotated[int, Query(..., description="Target cluster ID")],
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Move a face to a different cluster.

    Args:
        face_id: The face to move.
        cluster_id: Target cluster ID.

    Returns:
        Move confirmation.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        pipeline.db.client.set_payload(
            collection_name=pipeline.db.FACES_COLLECTION,
            payload={"cluster_id": cluster_id},
            points=[face_id],
        )
        return {"status": "moved", "face_id": face_id, "cluster_id": cluster_id}
    except Exception as e:
        logger.error(f"[Faces] Move failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
