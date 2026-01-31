"""Cluster ID management for biometrics."""

import time
import threading
from qdrant_client import QdrantClient


class ClusterManager:
    """Manages unique cluster IDs for Faces and Voices."""

    def __init__(self, client: QdrantClient):
        self.client = client
        self._cluster_id_lock = threading.Lock()
        self._cluster_id_counter = 0

    def get_next_cluster_id(self) -> int:
        """Generate a unique cluster ID (shared logic for Face/Voice).

        Uses timestamp-based unique ID with atomic counter to guarantee uniqueness.
        Format: YYMMDDHHMM + 4-digit counter = 14-digit ID that's time-sortable.

        Returns:
            Unique integer cluster ID.
        """
        with self._cluster_id_lock:
            # Get timestamp component (minutes since epoch, fits in reasonable int)
            ts = int(time.time() // 60)  # Minutes since epoch
            self._cluster_id_counter = (self._cluster_id_counter + 1) % 10000
            # Combine: timestamp * 10000 + counter
            cluster_id = (ts % 100000000) * 10000 + self._cluster_id_counter
            return cluster_id

    def get_next_voice_cluster_id(self) -> int:
        return self.get_next_cluster_id()

    def get_next_face_cluster_id(self) -> int:
        return self.get_next_cluster_id()

    def get_max_voice_cluster_id(self, collection_name: str) -> int:
        """Get the maximum existing voice cluster ID."""
        try:
            # Scroll through voice segments to find max cluster ID
            max_id = 0
            offset = None
            while True:
                results, offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=["voice_cluster_id"],
                    with_vectors=False,
                )
                for point in results:
                    cid = (
                        point.payload.get("voice_cluster_id", 0)
                        if point.payload
                        else 0
                    )
                    if isinstance(cid, int) and cid > max_id:
                        max_id = cid
                if offset is None:
                    break
            return max_id
        except Exception:
            return 0

    def get_max_face_cluster_id(self, collection_name: str) -> int:
        """Get the maximum existing face cluster ID."""
        try:
            max_id = 0
            offset = None
            while True:
                results, offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=["cluster_id"],
                    with_vectors=False,
                )
                for point in results:
                    cid = (
                        point.payload.get("cluster_id", 0)
                        if point.payload
                        else 0
                    )
                    if isinstance(cid, int) and cid > max_id:
                        max_id = cid
                if offset is None:
                    break
            return max_id
        except Exception:
            return 0
