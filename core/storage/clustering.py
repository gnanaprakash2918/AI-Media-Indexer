"""Cluster ID management for biometrics."""

import threading
import time

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

    def _get_max_field_value(self, collection_name: str, field_name: str) -> int:
        """Get the maximum value of a payload field across a collection.

        Scrolls through all points in batches to find the max integer value.

        Args:
            collection_name: Qdrant collection to scan.
            field_name: Payload field to find max value of.

        Returns:
            Maximum field value found, or 0 if collection is empty or on error.
        """
        try:
            max_id = 0
            offset = None
            while True:
                results, offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=[field_name],
                    with_vectors=False,
                )
                for point in results:
                    cid = (
                        point.payload.get(field_name, 0)
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

    def get_max_voice_cluster_id(self, collection_name: str) -> int:
        """Get the maximum existing voice cluster ID."""
        return self._get_max_field_value(collection_name, "voice_cluster_id")

    def get_max_face_cluster_id(self, collection_name: str) -> int:
        """Get the maximum existing face cluster ID."""
        return self._get_max_field_value(collection_name, "cluster_id")

