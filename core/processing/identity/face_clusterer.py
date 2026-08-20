"""Face clustering logic."""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from sklearn.cluster import HDBSCAN
from core.utils.logger import log
from core.utils.observe import observe
from typing import Sequence

class FaceClusterer:
    def __init__(self, eps=0.5, min_samples=3, metric="euclidean"):
        self.dbscan_eps = eps
        self.dbscan_min_samples = min_samples
        self.dbscan_metric = metric

    def cluster_faces(self, encodings: Sequence[ArrayLike]) -> NDArray[np.int64]:
        if not encodings:
            return np.array([], dtype=np.int64)
        
        arrs = [np.asarray(e, dtype=np.float64) for e in encodings]
        if len({a.shape for a in arrs}) > 1:
            raise ValueError("Inconsistent shapes")
        arr = np.vstack(arrs)
            
        return HDBSCAN(
            min_cluster_size=max(3, self.dbscan_min_samples),
            min_samples=self.dbscan_min_samples,
            cluster_selection_epsilon=self.dbscan_eps,
            metric=self.dbscan_metric,
            allow_single_cluster=True,
        ).fit_predict(arr)

    @observe("face_cluster")
    def match_or_create_cluster(
        self,
        embedding: list[float],
        existing_clusters: dict[int, list[float]],
        threshold: float = 0.4,
    ) -> tuple[int, dict[int, list[float]]]:
        emb = np.array(embedding, dtype=np.float64)
        emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
        best_id, best_sim = None, -1.0
        for cid, centroid in existing_clusters.items():
            cen_norm = np.array(centroid) / (np.linalg.norm(centroid) + 1e-9)
            sim = float(np.dot(emb_norm, cen_norm))
            if sim > (1.0 - threshold) and sim > best_sim:
                best_id, best_sim = cid, sim
        if best_id is not None and best_sim < 0.6:
            try:
                from core.processing.biometric_arbitrator import BIOMETRIC_ARBITRATOR
                if not BIOMETRIC_ARBITRATOR.should_merge_sync(
                    emb_norm,
                    np.array(existing_clusters[best_id]),
                    primary_sim=best_sim,
                ):
                    best_id = None
            except ImportError:
                pass
        if best_id is not None:
            new_cen = (np.array(existing_clusters[best_id]) + emb) / 2.0
            existing_clusters[best_id] = (
                new_cen / (np.linalg.norm(new_cen) + 1e-9)
            ).tolist()
            return best_id, existing_clusters
        nid = max(existing_clusters.keys(), default=0) + 1
        existing_clusters[nid] = emb_norm.tolist()
        return nid, existing_clusters

    async def resolve_identity_conflict(
        self,
        track_id: int,
        crop: NDArray[np.uint8],
        known: dict[int, NDArray[np.float32]],
    ) -> int:
        try:
            from core.processing.biometrics import get_biometric_arbitrator

            arb = get_biometric_arbitrator()
            emb = await arb.get_embedding(crop)
            if emb is None:
                return track_id
            if track_id in known and arb.verify_identity(emb, known[track_id]):
                return track_id
            mid = arb.find_matching_identity(emb, known)
            return mid if mid is not None else track_id
        except ImportError:
            return track_id
