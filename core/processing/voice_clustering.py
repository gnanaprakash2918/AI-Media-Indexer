"""Module for clustering speaker voices."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from config import settings
from core.storage.db import VectorDB
from core.utils.logger import log

# Handle HDBSCAN import with full type silence
try:
    from hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = Any


async def cluster_voices(db: VectorDB) -> dict[str, Any]:
    """Re-clusters all voice segments in the database using HDBSCAN.

    Groups similar voice embeddings into speaker clusters.
    Uses HDBSCAN with cosine distance for robust clustering, falling back
    to Agglomerative Clustering if noise levels are too high.

    Args:
        db: VectorDB instance.

    Returns:
        Clustering statistics.
    """
    # 1. Check requirements
    try:
        # Try hdbscan package first
        try:
            from hdbscan import HDBSCAN
        except ImportError:
            try:
                # Fallback to sklearn (newer versions 1.3+)
                from sklearn.cluster import HDBSCAN  # type: ignore
            except ImportError:
                # Last resort mock
                class HDBSCANStub:
                    def __init__(self, **kwargs):
                        pass

                    def fit_predict(self, X):
                        return np.full(X.shape[0], -1)

                HDBSCAN = HDBSCANStub

    except Exception as e:
        log(f"[VoiceClustering] Requirement check failed: {e}")
        return {"status": "error", "error": str(e)}

    log("[VoiceClustering] Starting global voice re-clustering...")

    # 2. Fetch all voice vectors
    try:
        points = []
        next_page = None
        while True:
            resp = db.client.scroll(
                collection_name=db.VOICE_COLLECTION,
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
            return {
                "status": "empty",
                "message": "No voice segments to cluster",
            }

        # Filter valid vectors
        valid_points = []
        vectors = []
        ids = []

        for p in points:
            vec = p.vector
            if not vec:
                continue
            if isinstance(vec, dict):
                vec = vec.get("vector") or vec.get("default")
            if not vec:
                continue
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()

            valid_points.append(p)
            vectors.append(vec)
            ids.append(str(p.id))

        if len(vectors) < 2:
            if len(vectors) == 1:
                # Assign single cluster 0
                db.client.set_payload(
                    collection_name=db.VOICE_COLLECTION,
                    points=[ids[0]],
                    payload={"cluster_id": 0},
                )
                return {"status": "success", "num_clusters": 1, "total": 1}
            return {"status": "skipped", "message": "Not enough samples"}

        # 3. Prepare Data
        X = np.array(vectors)

        # Normalize for cosine similarity (Voice embeddings benefit from this)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_normalized = X / (norms + 1e-8)

        # Compute Cosine Distance Matrix (1 - Similarity)
        distance_matrix = cosine_distances(X_normalized)

        # 4. Run HDBSCAN
        min_cluster = max(
            2, min(settings.hdbscan_min_cluster_size, len(vectors) // 3)
        )
        # Voice often has many small segments, so we allow smaller samples
        min_samples = 1

        hdb = HDBSCAN(
            min_cluster_size=min_cluster,
            min_samples=min_samples,
            cluster_selection_epsilon=settings.hdbscan_cluster_selection_epsilon,
            metric="precomputed",
            allow_single_cluster=True,
        )
        labels = hdb.fit_predict(distance_matrix)

        noise_count = list(labels).count(-1)
        noise_ratio = noise_count / len(labels)
        method = "HDBSCAN"

        # 5. Fallback: Agglomerative Clustering if too noisy
        if noise_ratio > 0.5:
            log(
                f"[VoiceClustering] High noise ({noise_ratio:.1%}). Fallback to Agglomerative."
            )
            agg = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=settings.face_clustering_threshold,  # Reuse threshold or add specific voice one
                metric="cosine",
                linkage="average",
            )
            labels = agg.fit_predict(X)
            method = "Agglomerative"
            noise_count = 0

        # 6. Update DB
        updates = 0
        for i, point_id in enumerate(ids):
            label = int(labels[i])
            cluster_id = label if label >= 0 else -1

            db.client.set_payload(
                collection_name=db.VOICE_COLLECTION,
                points=[point_id],
                payload={"cluster_id": cluster_id},
            )
            updates += 1

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        log(
            f"[VoiceClustering] Completed via {method}: {num_clusters} clusters from {len(vectors)} segments."
        )

        return {
            "status": "success",
            "total_segments": len(vectors),
            "num_clusters": num_clusters,
            "noise_points": noise_count,
            "method": method,
        }

    except Exception as e:
        log(f"[VoiceClustering] Failed: {e}")
        return {"status": "error", "error": str(e)}
