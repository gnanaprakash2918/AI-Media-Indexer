"""Face repository - all face CRUD and clustering operations.

Extracted from VectorDB to reduce God class size.
VectorDB inherits from FaceRepository to compose these methods.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from qdrant_client.http import models

from config import settings
from core.domain.values import ClusterId, Timestamp, VideoPath
from core.utils.logger import log
from core.storage.constants import (
    FACES_COLLECTION,
    MEDIA_COLLECTION
)


if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class FaceRepository:
    """Face detection, clustering, naming, and identity management operations."""

    def __init__(self, client: QdrantClient):
        self.client = client

    def get_next_face_cluster_id(self) -> int:
        """Generate a unique face cluster ID.

        Uses same logic as voice cluster IDs for consistency.

        Returns:
            Unique integer cluster ID.
        """
        return self.get_next_voice_cluster_id()

    def get_max_face_cluster_id(self) -> int:
        """Get the maximum existing face cluster ID."""
        try:
            max_id = 0
            offset = None
            while True:
                results, offset = self.client.scroll(
                    collection_name=FACES_COLLECTION,
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

    def upsert_face_cluster_centroid(
        self, cluster_id: int | ClusterId, embedding: list[float]
    ) -> None:
        """Stores or updates the centroid for a face cluster.

        Uses a deterministic ID based on cluster_id to allow easy retrieval/update.
        """
        import uuid

        # Deterministic UUID for the centroid
        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"face_centroid_{cluster_id}")
        )

        try:
            self.client.upsert(
                collection_name=FACES_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "face_cluster_id": cluster_id,
                            "is_centroid": True,
                            "type": "centroid",
                            "timestamp": time.time(),
                        },
                    )
                ],
            )
        except Exception as e:
            log(
                f"Failed to upsert face centroid {cluster_id}: {e}",
                level="ERROR",
            )

    def get_cluster_id_by_name(self, name: str) -> int | None:
        """Resolve a person's name to their cluster ID.

        Used by agentic search to filter frames by identity.

        Args:
            name: The person's name (from HITL naming).

        Returns:
            The cluster_id if found, None otherwise.
        """
        try:
            # Exact match search for name (case-sensitive)
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="name",
                            match=models.MatchValue(value=name),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            if resp[0] and resp[0][0].payload:
                return resp[0][0].payload.get("cluster_id")
            return None
        except Exception as e:
            log(
                f"get_cluster_id_by_name failed for '{name}': {e}",
                level="DEBUG",
            )
            return None

    def fuzzy_get_cluster_id_by_name(
        self, name: str, threshold: float = 0.7
    ) -> int | None:
        """Resolve a person's name to cluster ID using fuzzy matching.

        Handles:
        - Case-insensitive matching ("John" == "john")
        - Partial names ("Prakash" matches "Gnana Prakash")
        - Common variations ("Bob" might match "Robert")

        Args:
            name: The search name (can be partial or different case).
            threshold: Minimum similarity ratio (0.0-1.0) for a match.

        Returns:
            Best matching cluster_id, or None if no good match.
        """
        if not name or len(name) < 2:
            return None

        try:
            # Get all named faces
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                limit=1000,
                with_payload=["name", "cluster_id"],
                with_vectors=False,
            )
            best_match_id = None
            best_ratio = 0.0

            from difflib import SequenceMatcher

            search_lower = name.lower()

            for pt in resp[0]:
                payload = pt.payload or {}
                face_name = payload.get("name")
                if not face_name:
                    continue

                ratio = SequenceMatcher(
                    None, search_lower, face_name.lower()
                ).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match_id = payload.get("cluster_id")

            if best_ratio >= threshold:
                return best_match_id

            return None
        except Exception as e:
            log(
                f"fuzzy_get_cluster_id_by_name failed for '{name}': {e}",
                level="DEBUG",
            )
            return None

    def get_face_ids_by_cluster(self, cluster_id: int | ClusterId) -> list[str]:
        """Get all face point IDs belonging to a cluster.

        Args:
            cluster_id: The cluster ID to look up.

        Returns:
            List of face point IDs in that cluster.
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=1000,
                with_payload=False,
            )
            return [str(p.id) for p in resp[0]]
        except Exception as e:
            log(f"get_face_ids_for_video failed: {e}", level="DEBUG")
            return []

    def insert_face(
        self,
        face_encoding: list[float],
        name: str | None = None,
        cluster_id: int | None = None,
        media_path: str | None = None,
        timestamp: float | None = None,
        thumbnail_path: str | None = None,
        # Quality metrics for clustering (optional for backward compat)
        bbox_size: int | None = None,
        det_score: float | None = None,
        blur_score: float | None = None,
    ) -> str:
        """Insert a face embedding.

        Args:
            face_encoding: The numeric vector representing the face.
            name: Name of the person (if known).
            cluster_id: ID of the cluster this face belongs to.
            media_path: Source media file path.
            timestamp: Timestamp in the video where face was detected.
            thumbnail_path: Path to the face thumbnail image.
            bbox_size: Minimum dimension of face bounding box in pixels.
            det_score: Face detection confidence score.
            blur_score: Laplacian variance blur score (higher=sharper).

        Returns:
            The generated ID of the inserted point.
        """
        point_id = str(uuid.uuid4())
        # Auto-generate cluster_id from point_id hash if not provided
        if cluster_id is None:
            cluster_id = abs(hash(point_id)) % (10**9)

        payload = {
            "name": name,
            "cluster_id": cluster_id,
            "media_path": media_path,
            "timestamp": timestamp,
            "thumbnail_path": thumbnail_path,
            # Quality metrics
            "bbox_size": bbox_size,
            "det_score": det_score,
            "blur_score": blur_score,
        }

        self.client.upsert(
            collection_name=FACES_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=face_encoding,
                    payload=payload,
                )
            ],
        )

        return point_id

    def search_face(
        self,
        face_encoding: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar faces.

        Args:
            face_encoding: The query face vector.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.

        Returns:
            A list of matching faces.
        """
        resp = self.client.query_points(
            collection_name=FACES_COLLECTION,
            query=face_encoding,
            limit=limit,
            score_threshold=score_threshold,
        )

        results = []
        for hit in resp.points:
            payload = hit.payload or {}
            results.append(
                {
                    "score": hit.score,
                    "id": hit.id,
                    "name": payload.get("name"),
                    "cluster_id": payload.get("cluster_id"),
                }
            )

        return results

    def get_unresolved_faces(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get faces without assigned names.

        Args:
            limit: Maximum number of results.

        Returns:
            List of unnamed faces needing labeling.
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.IsNullCondition(
                            is_null=models.PayloadField(key="name")
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = []
            for point in resp[0]:
                payload = point.payload or {}
                cluster_id = payload.get("cluster_id")
                if cluster_id is None:
                    cluster_id = abs(hash(str(point.id))) % (10**9)
                results.append(
                    {
                        "id": point.id,
                        "cluster_id": cluster_id,
                        "name": payload.get("name"),
                        "media_path": payload.get("media_path"),
                        "timestamp": payload.get("timestamp"),
                        "thumbnail_path": payload.get("thumbnail_path"),
                        "is_main": payload.get("is_main", False),
                        "appearance_count": payload.get("appearance_count", 1),
                    }
                )
            # Sort: main characters first, then by appearance count
            results.sort(
                key=lambda x: (
                    not x.get("is_main", False),
                    -x.get("appearance_count", 1),
                )
            )
            return results
        except Exception:
            return []

    def update_face_name(self, cluster_id: int | ClusterId, name: str) -> int:
        """Assign a name to all faces in a cluster.

        Args:
            cluster_id: The cluster ID to update.
            name: The name to assign.

        Returns:
            Number of faces updated.
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=True,
            )
            points = resp[0]
            updated = 0
            for point in points:
                payload = point.payload or {}
                payload["name"] = name
                self.client.set_payload(
                    collection_name=FACES_COLLECTION,
                    payload=payload,
                    points=[point.id],
                )
                updated += 1

            # Propagate to media frames
            self._propagate_face_name_to_frames(cluster_id, name)

            return updated
        except Exception:
            return 0

    def update_face_cluster_id(
        self, face_id: str, cluster_id: int | ClusterId
    ) -> bool:
        """Update the cluster ID for a single face.

        Args:
            face_id: The ID of the face to update.
            cluster_id: The new cluster ID.

        Returns:
            True if updated successfully.
        """
        try:
            self.client.set_payload(
                collection_name=FACES_COLLECTION,
                payload={"cluster_id": cluster_id},
                points=[face_id],
            )
            return True
        except Exception as e:
            log("Failed to update face cluster ID", error=str(e))
            return False

    def merge_face_clusters(
        self, from_cluster: str | int, to_cluster: str | int
    ) -> int:
        """Merge all faces from one cluster into another.

        Args:
            from_cluster: Source cluster ID.
            to_cluster: Target cluster ID.

        Returns:
            Number of faces moved.
        """
        try:
            # First, check if the target cluster has a name
            target_name = None
            resp_target = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=to_cluster),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            if resp_target[0] and resp_target[0][0].payload:
                target_name = resp_target[0][0].payload.get("name")

            # Get all faces in source cluster
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=from_cluster),
                        )
                    ]
                ),
                limit=1000,
            )

            points = resp[0]
            if not points:
                return 0

            ids = [p.id for p in points]

            # Update cluster_id for all
            payload = {"cluster_id": to_cluster}
            # If target has a name, propagate it to the merged faces
            # (or if source had a name and target didn't, we might want to keep source name?
            # For now, let's assume target supersedes or we clear if ambiguous, but keeping target name is safer for HITL)
            if target_name:
                payload["name"] = target_name

            self.client.set_payload(
                collection_name=FACES_COLLECTION,
                payload=payload,
                points=ids,  # type: ignore
            )

            # Propagate cluster ID change to media frames
            self._update_frames_cluster_rename(from_cluster, to_cluster)

            return len(ids)
        except Exception:
            return 0

    def set_face_main(
        self, cluster_id: int | ClusterId, is_main: bool = True
    ) -> bool:
        """Set a face cluster as main character.

        This updates all faces in the cluster with is_main_character flag.
        Used by HITL to mark important recurring characters.

        Args:
            cluster_id: The face cluster ID to mark.
            is_main: Whether this is a main character.

        Returns:
            Success status.
        """
        try:
            # First get all face IDs in this cluster
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=1000,
                with_payload=False,
            )
            face_ids = [p.id for p in resp[0]]

            if not face_ids:
                log(f"[HITL] No faces found in cluster {cluster_id}")
                return False

            # Update all faces in cluster with PointIdsList (correct API usage)
            self.client.set_payload(
                collection_name=FACES_COLLECTION,
                payload={
                    "is_main_character": is_main,
                    "is_main": is_main,
                },  # Both keys for compat
                points=models.PointIdsList(points=face_ids),
            )
            log(
                f"[HITL] Set {len(face_ids)} faces in cluster {cluster_id} as main character: {is_main}"
            )
            return True
        except Exception as e:
            log(f"[HITL] Failed to set main character: {e}")
            return False

    def get_named_faces(self) -> list[dict[str, Any]]:
        """Get all named faces.

        Returns:
            List of named faces with their info.
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
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
            results = []
            for point in resp[0]:
                payload = point.payload or {}
                results.append(
                    {
                        "id": point.id,
                        "name": payload.get("name"),
                        "cluster_id": payload.get("cluster_id"),
                        "media_path": payload.get("media_path"),
                        "timestamp": payload.get("timestamp"),
                        "thumbnail_path": payload.get("thumbnail_path"),
                    }
                )
            return results
        except Exception:
            return []

    def delete_face(self, face_id: str) -> bool:
        """Delete a face by its ID.

        Args:
            face_id: The ID of the face to delete.

        Returns:
            True if deleted successfully, False otherwise.
        """
        try:
            self.client.delete(
                collection_name=FACES_COLLECTION,
                points_selector=models.PointIdsList(points=[face_id]),
            )
            return True
        except Exception:
            return False

    def update_single_face_name(self, face_id: str, name: str) -> bool:
        """Assign a name to a single face.

        Args:
            face_id: The ID of the face to update.
            name: The name to assign.

        Returns:
            True if updated successfully, False otherwise.
        """
        try:
            self.client.set_payload(
                collection_name=FACES_COLLECTION,
                payload={"name": name},
                points=[face_id],
            )
            return True
        except Exception:
            return False

    def get_faces_by_media(
        self, media_path: str | VideoPath, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Get all faces for a specific media file.

        Args:
            media_path: Path to the media file.
            limit: Maximum number of results.

        Returns:
            List of face data dicts.
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="media_path",
                            match=models.MatchValue(value=media_path),
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = []
            for point in resp[0]:
                payload = point.payload or {}
                results.append(
                    {
                        "id": point.id,
                        "media_path": payload.get("media_path"),
                        "timestamp": payload.get("timestamp"),
                        "name": payload.get("name"),
                        "cluster_id": payload.get("cluster_id"),
                        "thumbnail_path": payload.get("thumbnail_path"),
                    }
                )
            return results
        except Exception:
            return []

    def get_all_face_embeddings(self) -> list[dict[str, Any]]:
        """Get all face embeddings with their IDs for clustering.

        Returns:
            List of dicts with 'id' and 'embedding' keys.
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                limit=10000,
                with_payload=True,
                with_vectors=True,
            )
            results = []
            for point in resp[0]:
                if point.vector:
                    results.append(
                        {
                            "id": point.id,
                            "embedding": list(point.vector)
                            if isinstance(point.vector, (list, tuple))
                            else point.vector,
                            "payload": point.payload or {},
                        }
                    )
            return results
        except Exception:
            return []

    def get_faces_grouped_by_cluster(
        self, limit: int = 500
    ) -> dict[int, list[dict[str, Any]]]:
        """Get all faces grouped by cluster_id.

        Args:
            limit: Maximum number of faces to retrieve.

        Returns:
            Dictionary mapping cluster_id to list of faces in that cluster.
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            clusters: dict[int, list[dict[str, Any]]] = {}
            for point in resp[0]:
                payload = point.payload or {}
                cluster_id = payload.get("cluster_id", -1)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(
                    {
                        "id": point.id,
                        "name": payload.get("name"),
                        "cluster_id": cluster_id,
                        "media_path": payload.get("media_path"),
                        "timestamp": payload.get("timestamp"),
                        "thumbnail_path": payload.get("thumbnail_path"),
                    }
                )
            return clusters
        except Exception:
            return {}

    def get_all_cluster_centroids(self) -> dict[int, list[float]]:
        """Get cluster centroids for global identity matching.

        Returns ONE embedding per cluster_id (the mean of all faces in that cluster).
        This is O(1) per match instead of O(N) when matching new faces.

        Only returns clusters with at least one named face (HITL verified).
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                limit=10000,
                with_payload=True,
                with_vectors=True,
            )

            cluster_embeddings: dict[int, list[list[float]]] = {}
            cluster_names: dict[int, str | None] = {}

            for point in resp[0]:
                payload = point.payload or {}
                cluster_id = payload.get("cluster_id")
                name = payload.get("name")

                if cluster_id is None or point.vector is None:
                    continue

                if cluster_id not in cluster_embeddings:
                    cluster_embeddings[cluster_id] = []  # type: ignore
                    cluster_names[cluster_id] = name

                if name and not cluster_names[cluster_id]:
                    cluster_names[cluster_id] = name

                if isinstance(point.vector, list):
                    cluster_embeddings[cluster_id].append(point.vector)  # type: ignore
                elif hasattr(point.vector, "tolist"):
                    # Cast for Pylance safety or strict type check ignore
                    cluster_embeddings[cluster_id].append(point.vector.tolist())  # type: ignore
                elif isinstance(point.vector, dict):
                    # Handle named vectors - assuming we want the default or specific one
                    # If we don't know the name, we might skip or take values()
                    pass

            centroids: dict[int, list[float]] = {}
            for cluster_id, embeddings in cluster_embeddings.items():
                if embeddings:
                    import numpy as np

                    arr = np.array(embeddings, dtype=np.float64)
                    centroid = np.mean(arr, axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
                    centroids[cluster_id] = centroid.tolist()

            log(
                f"Loaded {len(centroids)} cluster centroids for global matching"
            )
            return centroids

        except Exception as e:
            log(f"Failed to get cluster centroids: {e}")
            return {}

    def update_cluster_centroid(
        self,
        cluster_id: int | ClusterId,
        new_embedding: list[float],
        alpha: float = 0.3,
    ) -> bool:
        """Update cluster centroid with exponential moving average.

        Args:
            cluster_id: Cluster to update.
            new_embedding: New face embedding to incorporate.
            alpha: EMA weight for new embedding (0.3 = 30% new, 70% old).
        """
        return True

    def delete_face_cluster(self, cluster_id: int | ClusterId) -> int:
        """Delete an entire face cluster and all its faces.

        Args:
            cluster_id: The face cluster ID to delete.

        Returns:
            Number of faces deleted.
        """
        try:
            # 1. Get all faces in this cluster
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            points = resp[0]
            if not points:
                return 0

            # 2. Delete thumbnail files
            for point in points:
                payload = point.payload or {}
                thumb_path = payload.get("thumbnail_path")
                if thumb_path:
                    try:
                        if thumb_path.startswith("/"):
                            file_path = settings.cache_dir / thumb_path.lstrip(
                                "/"
                            )
                            if file_path.exists():
                                file_path.unlink()
                    except Exception:
                        pass

            # 3. Delete the points
            point_ids = [point.id for point in points]
            self.client.delete(
                collection_name=FACES_COLLECTION,
                points_selector=models.PointIdsList(points=point_ids),
            )
            log(
                f"[DB] Deleted face cluster {cluster_id}: {len(point_ids)} faces"
            )
            return len(point_ids)
        except Exception as e:
            log(f"delete_face_cluster failed: {e}")
            return 0

    def get_face_by_thumbnail(
        self, thumbnail_path: str
    ) -> dict[str, Any] | None:
        """Look up a face by its thumbnail_path.

        Args:
            thumbnail_path: The thumbnail path stored in the database.

        Returns:
            Face data dict or None if not found.
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="thumbnail_path",
                            match=models.MatchValue(value=thumbnail_path),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if resp[0]:
                point = resp[0][0]
                payload = point.payload or {}
                return {
                    "id": point.id,
                    "media_path": payload.get("media_path"),
                    "timestamp": payload.get("timestamp", 0),
                    "name": payload.get("name"),
                    "cluster_id": payload.get("cluster_id"),
                    "thumbnail_path": payload.get("thumbnail_path"),
                }
            return None
        except Exception as e:
            log(f"get_face_by_thumbnail error: {e}")
            return None

    def get_face_name_by_cluster(self, cluster_id: str | int) -> str | None:
        """Get HITL-assigned name for a face cluster.

        Args:
            cluster_id: The face cluster ID.

        Returns:
            Name if assigned, None otherwise.
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=["name"],
            )
            if resp[0]:
                return (resp[0][0].payload or {}).get("name")
            return None
        except Exception:
            return None

    def get_face_cluster_by_name(self, name: str) -> int | None:
        """Find face cluster ID by HITL-assigned name.

        Used for auto-merging when naming a new cluster with an existing name.

        Args:
            name: The HITL-assigned name to search for.

        Returns:
            Cluster ID if found, None otherwise.
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="name",
                            match=models.MatchValue(value=name),
                        )
                    ]
                ),
                limit=1,
                with_payload=["cluster_id"],
            )
            if resp[0]:
                return (resp[0][0].payload or {}).get("cluster_id")
            return None
        except Exception:
            return None

    def set_face_name(self, cluster_id: str | int, name: str) -> int:
        """Set name for a face cluster (and all its points).

        Also propagates the name to all frames containing this cluster
        for proper search and display.

        **Auto-Merge Feature**: If another face cluster already has this name,
        both clusters will be merged under the same identity.

        **Cross-Modal Linking**: If a voice cluster has this same name,
        they will be linked together via the identity graph.

        Args:
            cluster_id: The face cluster ID (str or int).
            name: The name to assign.

        Returns:
            Number of updated face points.
        """
        try:
            # === Step 1: Check for existing face clusters with same name ===
            existing_cluster = self.get_face_cluster_by_name(name)
            if existing_cluster and existing_cluster != cluster_id:
                log(
                    f"[HITL] Found existing face cluster with name '{name}' (ID: {existing_cluster})"
                )
                log(
                    f"[HITL] Auto-merging cluster {cluster_id} into existing cluster {existing_cluster}"
                )

                # Merge current cluster into existing one
                try:
                    self.merge_face_clusters(
                        source_id=cluster_id, target_id=existing_cluster
                    )
                    # After merge, use the existing cluster for remaining operations
                    cluster_id = existing_cluster
                except Exception as merge_err:
                    log(
                        f"[HITL] Auto-merge failed, continuing with separate clusters: {merge_err}"
                    )

            # === Step 2: Get all face point IDs in this cluster ===
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=500,
            )
            point_ids = [str(p.id) for p in resp[0]]

            if not point_ids:
                log(f"set_face_name: No faces found for cluster {cluster_id}")
                return 0

            # === Step 3: Update all face points with the name ===
            self.client.set_payload(
                collection_name=FACES_COLLECTION,
                payload={"name": name},
                points=point_ids,  # type: ignore
            )

            # === Step 4: Propagate name to frames for proper search ===
            self._propagate_face_name_to_frames(cluster_id, name)

            # === Step 5: Identity Linking (Handled in SQL Repositories) ===
            pass

            log(
                f"[HITL] Set name '{name}' on {len(point_ids)} faces in cluster {cluster_id}"
            )
            return len(point_ids)
        except Exception as e:
            log(f"set_face_name failed: {e}")
            return 0

    def re_embed_face_cluster_frames(
        self, cluster_id: int | ClusterId, new_name: str
    ) -> int:
        """Update and re-embed all frames containing a face cluster after HITL naming."""
        try:
            updated = 0
            frames = self.get_frames_by_face_cluster(cluster_id)
            for frame in frames:
                frame_id = str(frame.get("id", ""))
                if not frame_id:
                    continue
                payload = frame.get("payload", {})
                face_names = list({*payload.get("face_names", []), new_name})
                speaker_names = payload.get("speaker_names", [])
                if self.update_frame_identity_text(
                    frame_id, face_names, speaker_names
                ):
                    updated += 1
            log(
                f"Re-embedded {updated} frames for face cluster {cluster_id} -> {new_name}"
            )
            return updated
        except Exception as e:
            log(f"re_embed_face_cluster_frames error: {e}")
            return 0

    def get_frames_by_face_cluster(
        self, cluster_id: int | ClusterId, limit: int = 1000
    ) -> list[dict]:
        """Get all frames containing a specific face cluster.

        Args:
            cluster_id: The face cluster ID to search for.
            limit: Maximum results.

        Returns:
            List of frame data dicts.
        """
        try:
            resp = self.client.scroll(
                collection_name=MEDIA_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="face_cluster_ids",
                            match=models.MatchAny(any=[cluster_id]),
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            results = []
            for point in resp[0]:
                results.append(
                    {
                        "id": str(point.id),
                        "payload": point.payload or {},
                        "vector": point.vector,
                    }
                )
            return results
        except Exception as e:
            log(f"get_frames_by_face_cluster error: {e}")
            return []

    def create_empty_face_cluster(
        self, cluster_id: str, name: str = "", source: str = "manual"
    ) -> bool:
        """Creates a placeholder face cluster entry.

        Used for manual identity initialization or HITL workflows.

        Args:
            cluster_id: The cluster identifier to create.
            name: Optional name to assign to the cluster.
            source: The source of the cluster creation ('manual', 'auto').

        Returns:
            True if created successfully, False otherwise.
        """
        import numpy as np

        dummy_vector = np.zeros(512).tolist()
        point_id = str(uuid.uuid4())
        try:
            self.client.upsert(
                collection_name=FACES_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=dummy_vector,
                        payload={
                            "cluster_id": cluster_id,
                            "name": name,
                            "source": source,
                            "verified": False,
                            "is_placeholder": True,
                        },
                    )
                ],
            )
            return True
        except Exception as e:
            log(f"create_empty_face_cluster failed: {e}")
            return False

    def move_face_to_cluster(
        self, face_id: str, target_cluster_id: str
    ) -> bool:
        """Moves a single face point from its current cluster to a target cluster.

        Args:
            face_id: The point ID of the face to move.
            target_cluster_id: The ID of the destination cluster.

        Returns:
            True if the move was successful, False otherwise.
        """
        try:
            self.client.set_payload(
                collection_name=FACES_COLLECTION,
                payload={"cluster_id": target_cluster_id},
                points=[face_id],
            )
            return True
        except Exception as e:
            log(f"move_face_to_cluster failed: {e}")
            return False

    def recalculate_cluster_centroid(
        self, cluster_id: str | int
    ) -> list[float] | None:
        """Computes the mean embedding vector (centroid) for a face cluster.

        Filters out placeholder points and aggregates real face embeddings.

        Args:
            cluster_id: The cluster ID to recalculate.

        Returns:
            The mean vector as a list of floats, or None if no vectors found.
        """
        import numpy as np

        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=100,
                with_vectors=True,
            )
            if not resp[0]:
                return None

            # Extract vectors robustly
            vectors = []
            for p in resp[0]:
                if not (p.payload or {}).get("is_placeholder"):
                    if p.vector:
                        if isinstance(p.vector, list):
                            vectors.append(p.vector)
                        elif (
                            isinstance(p.vector, dict) and "vector" in p.vector
                        ):
                            vectors.append(p.vector["vector"])

            if not vectors:
                return None
            centroid = np.mean(vectors, axis=0).tolist()
            return centroid
        except Exception as e:
            log(f"recalculate_cluster_centroid failed: {e}")
            return None

    def get_cluster_distance(
        self, source_id: str | int, target_id: str | int
    ) -> float | None:
        """Calculates the cosine distance between two cluster centroids.

        Args:
            source_id: First cluster ID.
            target_id: Second cluster ID.

        Returns:
            Cosine distance (0.0 to 1.0) or None if calculation fails.
        """
        import numpy as np

        try:
            src_centroid = self.recalculate_cluster_centroid(source_id)
            tgt_centroid = self.recalculate_cluster_centroid(target_id)
            if src_centroid is None or tgt_centroid is None:
                return None
            src_arr = np.array(src_centroid)
            tgt_arr = np.array(tgt_centroid)
            dist = 1.0 - np.dot(src_arr, tgt_arr) / (
                np.linalg.norm(src_arr) * np.linalg.norm(tgt_arr) + 1e-8
            )
            return float(dist)
        except Exception:
            return None

    def _update_frames_cluster_rename(
        self, old_cluster: str | int, new_cluster: str | int
    ) -> int:
        """Updates face cluster references in media frames after a merge or rename.

        Args:
            old_cluster: The previous cluster ID.
            new_cluster: The new cluster ID to replace it with.

        Returns:
            The number of frames updated.
        """
        try:
            resp = self.client.scroll(
                collection_name=MEDIA_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="face_cluster_ids",
                            match=models.MatchAny(any=[old_cluster]),  # type: ignore
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
            )
            updated = 0
            for p in resp[0]:
                payload = p.payload or {}
                clusters = payload.get("face_cluster_ids", [])
                new_clusters = [
                    new_cluster if c == old_cluster else c for c in clusters
                ]
                self.client.set_payload(
                    collection_name=MEDIA_COLLECTION,
                    payload={"face_cluster_ids": new_clusters},
                    points=[str(p.id)],
                )
                updated += 1
            return updated
        except Exception:
            return 0

    def set_cluster_verified(
        self, cluster_id: str | int, verified: bool = True
    ) -> bool:
        """Marks a face cluster as HITL-verified.

        Updates the 'verified' flag for all face points in the cluster.

        Args:
            cluster_id: The cluster ID to verify.
            verified: The verification status to set.

        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            resp = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=500,
            )
            point_ids = [str(p.id) for p in resp[0]]
            if point_ids:
                self.client.set_payload(
                    collection_name=FACES_COLLECTION,
                    payload={"verified": verified},
                    points=point_ids,  # type: ignore
                )
            return True
        except Exception:
            return False

    def _propagate_face_name_to_frames(
        self, cluster_id: str | int, name: str
    ) -> int:
        """Updates 'face_names' list in media frames for a specific cluster.

        Ensures that when a cluster is named, all associated frames reflecting
        that cluster's presence have the name in their metadata for search.

        Args:
            cluster_id: The ID of the face cluster.
            name: The name to propagate.

        Returns:
            The number of frames updated.
        """
        try:
            resp = self.client.scroll(
                collection_name=MEDIA_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="face_cluster_ids",
                            match=models.MatchAny(any=[cluster_id]),  # type: ignore
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
            )
            updated = 0
            for p in resp[0]:
                payload = p.payload or {}
                names = list({*payload.get("face_names", []), name})
                self.client.set_payload(
                    collection_name=MEDIA_COLLECTION,
                    payload={"face_names": names},
                    points=[str(p.id)],
                )
                updated += 1
            return updated
        except Exception:
            return 0

    def _merge_face_cluster_frames(self, source_id: int, target_id: int):
        """Helper to update frame references when merging face clusters."""
        try:
            # Scroll frames that have source_id
            resp = self.client.scroll(
                collection_name=MEDIA_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="face_cluster_ids",
                            match=models.MatchAny(any=[source_id]),
                        )
                    ]
                ),
                limit=10000,
                with_payload=["face_cluster_ids"],
            )

            for p in resp[0]:
                payload = p.payload or {}
                ids = payload.get("face_cluster_ids", [])
                if source_id in ids:
                    new_ids = [target_id if x == source_id else x for x in ids]
                    # Deduplicate
                    new_ids = list(set(new_ids))
                    self.client.set_payload(
                        collection_name=MEDIA_COLLECTION,
                        payload={"face_cluster_ids": new_ids},
                        points=[p.id],
                    )
        except Exception as e:
            log(f"_merge_face_cluster_frames failed: {e}")

    async def get_faces_in_range(
        self,
        media_path: str | VideoPath,
        start_time: float | Timestamp,
        end_time: float | Timestamp,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get face detections in a time range for a specific video.

        Used by agentic_search to enrich search results with face data.
        """
        try:
            conditions = [
                models.FieldCondition(
                    key="media_path",
                    match=models.MatchValue(value=media_path),
                ),
                models.FieldCondition(
                    key="timestamp",
                    range=models.Range(gte=start_time, lte=end_time),
                ),
            ]
            query_filter = models.Filter(must=conditions)

            results, _ = self.client.scroll(
                collection_name=FACES_COLLECTION,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

            faces = []
            for point in results:
                payload = point.payload or {}
                faces.append(
                    {
                        "id": str(point.id),
                        "name": payload.get("name"),
                        "cluster_id": payload.get("cluster_id"),
                        "timestamp": payload.get("timestamp"),
                        "bbox": payload.get("bbox"),
                        "bbox_size": payload.get("bbox_size"),
                        "det_score": payload.get("det_score"),
                        "media_path": payload.get("media_path"),
                    }
                )
            return faces
        except Exception as e:
            log(f"get_faces_in_range failed: {e}")
            return []
