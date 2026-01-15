"""Module for clustering faces and other biometrics."""

import asyncio
from typing import Any

from config import settings
from core.storage.db import VectorDB
from core.utils.logger import log


async def cluster_faces(db: VectorDB) -> dict[str, Any]:
    """Re-clusters all faces in the database using HDBSCAN.

    This is a heavy operation that:
    1. Fetches all face vectors (including those with assigned names if needed, 
       though usually we only cluster unnamed ones or re-cluster everything).
    2. Runs HDBSCAN.
    3. Updates cluster_ids in Qdrant.
    
    Args:
        db: The VectorDB instance.

    Returns:
        Statistics dictionary {total_faces, num_clusters, noise_points}.
    """
    try:
        from sklearn.cluster import HDBSCAN
        import numpy as np
        from qdrant_client.http import models
    except ImportError as e:
        log(f"[Clustering] Failed to import requirements: {e}")
        return {"status": "failed", "error": str(e)}

    log("[Clustering] Starting global face re-clustering...")

    # 1. Fetch all face vectors
    try:
        # Scroll all points from "faces" collection
        points = []
        next_page = None
        while True:
            resp = db.client.scroll(
                collection_name=db.FACES_COLLECTION,
                limit=1000,
                offset=next_page,
                with_vectors=True,
                with_payload=True,
            )
            batch, next_page = resp
            points.extend(batch)
            if not next_page:
                break
        
        if not points:
            return {"status": "empty", "message": "No faces to cluster"}

        # Filter out placeholders or invalid vectors
        valid_points = []
        vectors = []
        ids = []

        for p in points:
            vec = p.vector
            if not vec:
                continue
            # Handle list vs dict vector (Qdrant)
            if isinstance(vec, dict):
                vec = vec.get("vector") or vec.get("default")
            
            if not vec:
                continue
            
            # Ensure it's a list
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()
            
            valid_points.append(p)
            vectors.append(vec)
            ids.append(str(p.id))

        if len(vectors) < 2:
            return {"status": "skipped", "message": "Not enough faces to cluster"}

        X = np.array(vectors)

        # 2. Run HDBSCAN
        # Use settings from config.py
        hdb = HDBSCAN(
            min_cluster_size=settings.hdbscan_min_cluster_size,
            min_samples=settings.hdbscan_min_samples,
            cluster_selection_epsilon=settings.hdbscan_cluster_selection_epsilon,
            metric="euclidean", # Cosine distance is usually 1-cosine similarity. 
                                # If vectors are normalized, euclidean is proportional to cosine.
                                # But HDBSCAN with euclidean on normalized vectors is standard approx.
            allow_single_cluster=True,
        )
        labels = hdb.fit_predict(X)

        # 3. Update Cluster IDs
        updates = 0
        
        # Map labels to cluster IDs. 
        # HDBSCAN returns -1 for noise.
        # We want persistent cluster IDs? 
        # Ideally, we should try to match new clusters to old meaningful IDs if possible, 
        # but for a full re-cluster, we might wipe and reset or just use the label ID.
        # For simplicity in this fix, we map label X -> cluster_id X.
        # "Unknown" (noise) -> -1 or specific ID.
        
        for i, point_id in enumerate(ids):
            label = int(labels[i])
            cluster_id = label if label >= 0 else -1
            
            db.client.set_payload(
                collection_name=db.FACES_COLLECTION,
                points=[point_id],
                payload={"cluster_id": cluster_id}
            )
            updates += 1

        log(f"[Clustering] Completed. Processed {len(vectors)} faces into {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
        
        return {
            "total_faces": len(vectors),
            "num_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "noise_points": list(labels).count(-1),
            "params": {
                "min_cluster_size": settings.hdbscan_min_cluster_size,
                "min_samples": settings.hdbscan_min_samples,
                "epsilon": settings.hdbscan_cluster_selection_epsilon
            }
        }

    except Exception as e:
        log(f"[Clustering] Error: {e}")
        return {"status": "error", "error": str(e)}
