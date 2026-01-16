"""Module for clustering faces and other biometrics."""

from typing import Any

from config import settings
from core.storage.db import VectorDB
from core.utils.logger import log

# Handle HDBSCAN import with full type silence to satisfy Pylance
try:
    from hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = Any


async def cluster_faces(db: VectorDB) -> dict[str, Any]:
    """Re-clusters all faces in the database using HDBSCAN.

    This is a heavy operation that:
    1. Fetches all face vectors (including those with assigned names if needed,
       though usually we only cluster unnamed ones or re-cluster everything).
    2. Runs HDBSCAN.
    3. If >50% noise, falls back to Agglomerative Clustering.
    4. Updates cluster_ids in Qdrant.

    Args:
        db: The VectorDB instance.

    Returns:
        Statistics dictionary {total_faces, num_clusters, noise_points}.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering

        # Try hdbscan package first (preferred for this project likely)
        try:
            from hdbscan import HDBSCAN
        except ImportError:
            try:
                # Fallback to sklearn (newer versions 1.3+)
                from sklearn.cluster import HDBSCAN  # type: ignore
            except ImportError:
                # Last resort mock for typing
                class HDBSCANStub:
                    def __init__(
                        self,
                        min_cluster_size: int = 5,
                        min_samples: int | None = None,
                        cluster_selection_epsilon: float = 0.0,
                        metric: str = "euclidean",
                        allow_single_cluster: bool = False,
                    ):
                        pass

                    def fit_predict(self, X: Any) -> Any:  # noqa: N803
                        return np.full(X.shape[0], -1)

        import numpy as np
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
            # Special case: only 1 face, assign cluster 0
            if len(vectors) == 1:
                db.client.set_payload(
                    collection_name=db.FACES_COLLECTION,
                    points=[ids[0]],
                    payload={"cluster_id": 0},
                )
                return {
                    "status": "success",
                    "total_faces": 1,
                    "num_clusters": 1,
                    "noise_points": 0,
                }
            return {
                "status": "skipped",
                "message": "Not enough faces to cluster",
            }

        X = np.array(vectors)  # noqa: N806

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_normalized = X / (norms + 1e-8)  # noqa: N806

        # 2. Run HDBSCAN first
        # Use settings from config.py with COSINE distance (better for face embeddings)
        # HDBSCAN doesn't directly support cosine, so we compute cosine distance matrix
        # Cosine distance = 1 - cosine_similarity
        from sklearn.metrics.pairwise import cosine_distances

        distance_matrix = cosine_distances(X_normalized)

        # For small datasets, use more lenient parameters
        min_cluster = max(
            2, min(settings.hdbscan_min_cluster_size, len(vectors) // 3)
        )
        min_samples = (
            1  # Allow single samples as core points for small datasets
        )

        hdb = HDBSCAN(
            min_cluster_size=min_cluster,
            min_samples=min_samples,
            cluster_selection_epsilon=settings.hdbscan_cluster_selection_epsilon,
            metric="precomputed",  # Use our cosine distance matrix
            allow_single_cluster=True,
        )
        labels = hdb.fit_predict(distance_matrix)

        # Count noise points
        noise_count = list(labels).count(-1)
        noise_ratio = noise_count / len(labels)
        clustering_method = "HDBSCAN"

        # 3. FALLBACK: If >50% are noise, use Agglomerative Clustering
        if noise_ratio > 0.5:
            log(
                f"[Clustering] HDBSCAN produced {noise_ratio * 100:.1f}% noise ({noise_count}/{len(labels)}). Using Agglomerative Clustering fallback."
            )

            # Use Agglomerative Clustering with cosine distance
            # n_clusters=None means we use distance_threshold
            agg = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=settings.face_clustering_threshold,  # 0.5 = 50% similarity
                metric="cosine",
                linkage="average",
            )
            labels = agg.fit_predict(X)
            clustering_method = "Agglomerative"
            noise_count = 0  # Agglomerative doesn't produce noise

            log(
                f"[Clustering] Agglomerative produced {len(set(labels))} clusters"
            )

        # 4. Update Cluster IDs in DB
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
                payload={"cluster_id": cluster_id},
            )
            updates += 1

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        log(
            f"[Clustering] Completed using {clustering_method}. Processed {len(vectors)} faces into {num_clusters} clusters."
        )

        return {
            "total_faces": len(vectors),
            "num_clusters": num_clusters,
            "noise_points": noise_count,
            "method": clustering_method,
            "params": {
                "min_cluster_size": settings.hdbscan_min_cluster_size,
                "min_samples": settings.hdbscan_min_samples,
                "epsilon": settings.hdbscan_cluster_selection_epsilon,
            },
        }

    except Exception as e:
        log(f"[Clustering] Error: {e}")
        return {"status": "error", "error": str(e)}
