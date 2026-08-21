"""Voice repository - all voice/speaker CRUD and diarization operations.

Extracted from VectorDB to reduce God class size.
VectorDB inherits from VoiceRepository to compose these methods.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from qdrant_client.http import models

from config import settings
from core.domain.values import ClusterId, Timestamp, VideoPath
from core.utils.logger import log
from core.storage.constants import (
    MEDIA_COLLECTION,
    VOICE_COLLECTION,
    VOICE_VECTOR_SIZE
)


if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class VoiceRepository:
    """Voice segment, speaker clustering, naming, and diarization operations."""

    def __init__(self, client: QdrantClient):
        self.client = client

    def get_max_voice_cluster_id(self) -> int:
        """Get the maximum existing voice cluster ID.

        Useful for bootstrapping counter after restart.

        Returns:
            Maximum cluster ID found, or 0 if none exist.
        """
        try:
            # Scroll through voice segments to find max cluster ID
            max_id = 0
            offset = None
            while True:
                results, offset = self.client.scroll(
                    collection_name=VOICE_COLLECTION,
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

    def get_voice_segments_by_video(
        self,
        video_path: str | VideoPath,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict]:
        """Retrieves voice diarization segments for a video.

        Used by the overlays API for speaker timeline visualization.

        Args:
            video_path: Path string of the target video.
            start_time: Optional start time filter (seconds).
            end_time: Optional end time filter (seconds).

        Returns:
            List of voice segment dictionaries with speaker info.
        """
        segments = []

        # Build filter conditions
        must_conditions = [
            models.FieldCondition(
                key="media_path",
                match=models.MatchValue(value=video_path),
            )
        ]

        if start_time is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="start",
                    range=models.Range(gte=start_time),
                )
            )
        if end_time is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="end",
                    range=models.Range(lte=end_time),
                )
            )

        try:
            results, _ = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=models.Filter(must=must_conditions),
                limit=5000,
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                payload = point.payload
                segments.append(
                    {
                        "start": payload.get("start", 0),
                        "end": payload.get("end", 0),
                        "speaker_id": payload.get("speaker_label", ""),
                        "speaker_name": payload.get("speaker_name", ""),
                        "cluster_id": payload.get("voice_cluster_id", -1),
                        "transcript": payload.get("transcript", ""),
                    }
                )
        except Exception as e:
            log(f"Failed to get voice segments: {e}")

        segments.sort(key=lambda x: x.get("start", 0))
        return segments

    async def match_speaker(
        self,
        embedding: list[float],
        threshold: float = 0.5,
    ) -> tuple[str, int, float] | None:
        """Finds a matching global speaker identity for a given embedding.

        Performs a nearest-neighbor search in the voice collection.

        Args:
            embedding: The 256-dim voice embedding vector.
            threshold: Similarity threshold for considering a match.

        Returns:
            A tuple of (speaker_id, cluster_id, score) if a match is found, else None.
        """
        try:
            resp = self.client.query_points(
                collection_name=VOICE_COLLECTION,
                query=embedding,
                limit=1,
                score_threshold=threshold,
            )
            if resp.points:
                hit = resp.points[0]
                payload = hit.payload or {}
                return (
                    payload.get("speaker_id", "unknown"),
                    payload.get("voice_cluster_id", -1),
                    hit.score,
                )
        except Exception as e:
            log(f"match_speaker failed: {e}", level="DEBUG")
        return None

    def upsert_voice_cluster_centroid(
        self, cluster_id: int | ClusterId, embedding: list[float]
    ) -> None:
        """Stores or updates the centroid for a voice cluster.

        Uses a deterministic ID based on cluster_id to allow easy retrieval/update.
        """
        import time
        import uuid

        # Deterministic UUID for the centroid
        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"voice_centroid_{cluster_id}")
        )

        try:
            self.client.upsert(
                collection_name=VOICE_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "voice_cluster_id": cluster_id,
                            "is_centroid": True,
                            "type": "centroid",
                            "timestamp": time.time(),
                            "speaker_id": f"CLUSTER_{cluster_id}",  # consistent ID
                        },
                    )
                ],
            )
        except Exception as e:
            log(
                f"Failed to upsert voice centroid {cluster_id}: {e}",
                level="ERROR",
            )

    def upsert_speaker_embedding(
        self,
        speaker_id: str,
        embedding: list[float],
        media_path: str | VideoPath,
        start: float,
        end: float,
        voice_cluster_id: int = -1,
    ) -> None:
        """Stores a voice segment embedding linked to a global speaker ID.

        Args:
            speaker_id: Unique identifier for the speaker.
            embedding: The voice embedding vector.
            media_path: Path to the source media file.
            start: Start timestamp of the voice segment.
            end: End timestamp of the voice segment.
            voice_cluster_id: The cluster ID this speaker belongs to.
        """
        import uuid

        point_id = str(uuid.uuid4())

        self.client.upsert(
            collection_name=VOICE_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "speaker_id": speaker_id,
                        "media_path": media_path,
                        "start": start,
                        "end": end,
                        "type": "voice_sample",
                        "voice_cluster_id": voice_cluster_id,
                    },
                )
            ],
        )

    def insert_voice_segment(
        self,
        *,
        media_path: str | VideoPath,
        start: float,
        end: float,
        speaker_label: str,
        embedding: list[float],
        audio_path: str | None = None,
        voice_cluster_id: int = -1,
        **kwargs: Any,
    ) -> None:
        """Insert a voice segment embedding.

        Args:
            media_path: Path to the source media file.
            start: Start time of the segment.
            end: End time of the segment.
            speaker_label: Label or ID of the speaker.
            embedding: The voice embedding vector.
            audio_path: Path to the extracted audio clip.
            voice_cluster_id: The cluster ID for grouping (-1 = unclustered).
            **kwargs: Additional metadata to store in payload (e.g. emotion).

        Raises:
            ValueError: If the embedding dimension does not match `VOICE_VECTOR_SIZE`.
        """
        if len(embedding) != VOICE_VECTOR_SIZE:
            raise ValueError(
                f"voice vector dim mismatch: expected {VOICE_VECTOR_SIZE}, "
                f"got {len(embedding)}"
            )

        # Deterministic ID to prevent duplicates (idempotency)
        unique_str = f"voice_{media_path}_{start:.3f}_{end:.3f}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

        payload = {
            "media_path": media_path,
            # Standardized keys (best practice for cross-modal fusion)
            "start_time": start,
            "end_time": end,
            # Legacy keys for backwards compatibility
            "start": start,
            "end": end,
            "speaker_label": speaker_label,
            "embedding_version": "wespeaker_resnet34_v1_l2",
            "audio_path": audio_path,
            "voice_cluster_id": voice_cluster_id,
        }
        if kwargs:
            payload.update(kwargs)

        self.client.upsert(
            collection_name=VOICE_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

    def set_speaker_name(self, cluster_id: int | ClusterId, name: str) -> int:
        """Assign a name to a voice cluster.

        Args:
            cluster_id: The voice cluster ID.
            name: The user-assigned name.

        Returns:
            Number of segments updated.
        """
        try:
            self.client.set_payload(
                collection_name=VOICE_COLLECTION,
                payload={"speaker_name": name},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="voice_cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
            )
            # Find how many points were updated (optional, for return value)
            # For speed, we can skip counting or do a quick separate count query.
            # Let's just return 1 to indicate success for now or perform a count.
            return 1
        except Exception as e:
            log(f"set_speaker_name failed: {e}")
            return 0

    def set_speaker_main(
        self, cluster_id: int | ClusterId, segment_id: str, is_main: bool = True
    ) -> bool:
        """Mark a specific segment as the 'main' representation of a speaker.

        Args:
            cluster_id: The voice cluster ID.
            segment_id: The specific segment ID to mark.
            is_main: Boolean status.

        Returns:
            Success status.
        """
        try:
            # 1. Unmark others in cluster if setting to True (enforce single main?)
            # Usually we allow multiple mains or just one. Assuming one main per cluster.
            if is_main:
                self.client.set_payload(
                    collection_name=VOICE_COLLECTION,
                    payload={"is_main": False},
                    points=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="voice_cluster_id",
                                match=models.MatchValue(value=cluster_id),
                            )
                        ]
                    ),
                )

            # 2. Set target
            self.client.set_payload(
                collection_name=VOICE_COLLECTION,
                payload={"is_main": is_main},
                points=models.PointIdsList(points=[segment_id]),
            )
            return True
        except Exception as e:
            log(f"set_speaker_main failed: {e}")
            return False

    async def search_voice_segments(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        video_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search voice segments using keyword/substring matching.

        NOTE: Semantic vector search is disabled for voice segments because the
        collection currently stores speaker embeddings (256d), not text embeddings (1024d).
        Unlocks 'search by transcript' functionality via text matching.

        Args:
            query: The search query (text or speaker name).
            limit: Maximum number of results.
            score_threshold: Ignored for keyword search (compatibility param).
            video_path: Optional filter by video path.

        Returns:
            List of matching voice segments.
        """
        try:
            # Build filter conditions for Keyword Search
            should_conditions = []

            # 1. Search in transcription text
            should_conditions.append(
                models.FieldCondition(
                    key="text", match=models.MatchText(text=query)
                )
            )
            should_conditions.append(
                models.FieldCondition(
                    key="transcription", match=models.MatchText(text=query)
                )
            )

            # 2. Search in speaker name
            should_conditions.append(
                models.FieldCondition(
                    key="speaker_name", match=models.MatchText(text=query)
                )
            )

            # 3. Search in speaker_id (exact match or partial)
            should_conditions.append(
                models.FieldCondition(
                    key="speaker_id", match=models.MatchText(text=query)
                )
            )

            must_conditions = []
            if video_path:
                must_conditions.append(
                    models.FieldCondition(
                        key="media_path",
                        match=models.MatchValue(value=video_path),
                    )
                )

            query_filter = models.Filter(
                should=should_conditions,
                must=must_conditions if must_conditions else None,
            )

            # Use Scroll (no vector scoring)
            results, _ = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            formatted_results = []
            for hit in results:
                payload = hit.payload or {}
                # Extract timestamps with fallback for legacy data
                ts_start = payload.get("start_time") or payload.get("start", 0)
                ts_end = payload.get("end_time") or payload.get("end", 0)
                formatted_results.append(
                    {
                        "id": str(hit.id),
                        "score": 1.0,  # distinct from vector score
                        "type": "voice_segment",
                        "speaker_id": payload.get("speaker_id"),
                        "speaker_name": payload.get(
                            "speaker_name", "Unknown Speaker"
                        ),
                        "text": payload.get(
                            "text", payload.get("transcription", "")
                        ),
                        # Standardized keys (used by fusion)
                        "start_time": ts_start,
                        "end_time": ts_end,
                        # Legacy keys for backwards compatibility
                        "start": ts_start,
                        "end": ts_end,
                        "video_path": payload.get("media_path"),
                        "media_path": payload.get(
                            "media_path"
                        ),  # Also standardize path
                        **payload,
                    }
                )
            return formatted_results
        except Exception as e:
            log(f"search_voice_segments failed: {e}")
            return []

    def get_voice_segments(
        self,
        media_path: str | None = None,
        emotion: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get voice segments, optionally filtered by media path or emotion.

        Args:
            media_path: Optional filter by media file path.
            emotion: Optional filter by emotion (e.g., "happy", "sad", "angry").
            limit: Maximum number of results.

        Returns:
            List of voice segments.
        """
        try:
            conditions = []
            if media_path:
                conditions.append(
                    models.FieldCondition(
                        key="media_path",
                        match=models.MatchValue(value=media_path),
                    )
                )
            if emotion:
                conditions.append(
                    models.FieldCondition(
                        key="emotion",
                        match=models.MatchValue(value=emotion),
                    )
                )
            qfilter = models.Filter(must=conditions) if conditions else None
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=qfilter,
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
                        "start": payload.get("start"),
                        "end": payload.get("end"),
                        "start_time": payload.get("start_time")
                        or payload.get("start"),
                        "end_time": payload.get("end_time")
                        or payload.get("end"),
                        "speaker_label": payload.get("speaker_label"),
                        "audio_path": payload.get("audio_path"),
                        "emotion": payload.get("emotion"),
                        "emotion_conf": payload.get("emotion_conf"),
                        "voice_cluster_id": payload.get("voice_cluster_id"),
                    }
                )
            return results
        except Exception:
            return []

    def delete_voice_segment(self, segment_id: str) -> bool:
        """Delete a voice segment and its audio file.

        Args:
            segment_id: The ID of the segment.

        Returns:
            True if deleted successfully.
        """
        try:
            # 1. Get payload to find file
            points = self.client.retrieve(
                collection_name=VOICE_COLLECTION,
                ids=[segment_id],
                with_payload=True,
            )

            if points:
                payload = points[0].payload or {}
                audio = payload.get("audio_path")
                if audio:
                    try:
                        if audio.startswith("/thumbnails"):
                            path = settings.cache_dir / audio.lstrip("/")
                            if path.exists():
                                path.unlink()
                    except Exception:
                        pass

            # 2. Delete point
            self.client.delete(
                collection_name=VOICE_COLLECTION,
                points_selector=models.PointIdsList(points=[segment_id]),
            )
            return True
        except Exception:
            return False

    def update_voice_speaker_name(self, segment_id: str, name: str) -> bool:
        """Update the speaker name for a voice segment.

        Args:
            segment_id: The ID of the voice segment.
            name: The human-readable speaker name.

        Returns:
            True if updated successfully.
        """
        try:
            self.client.set_payload(
                collection_name=VOICE_COLLECTION,
                payload={"speaker_name": name},
                points=[segment_id],
            )
            return True
        except Exception:
            return False

    def get_all_voice_embeddings(self) -> list[dict[str, Any]]:
        """Get all voice embeddings with their IDs for clustering.

        Returns:
            List of dicts with 'id', 'embedding', and 'payload' keys.
        """
        try:
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
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

    def update_voice_cluster_id(
        self, segment_id: str, cluster_id: int | ClusterId
    ) -> bool:
        """Update the voice_cluster_id for a voice segment.

        Args:
            segment_id: The ID of the voice segment.
            cluster_id: The new cluster ID.

        Returns:
            True if updated successfully.
        """
        try:
            self.client.set_payload(
                collection_name=VOICE_COLLECTION,
                payload={"voice_cluster_id": cluster_id},
                points=[segment_id],
            )
            return True
        except Exception:
            return False

    def get_voices_grouped_by_cluster(
        self, limit: int = 500
    ) -> dict[int, list[dict[str, Any]]]:
        """Get all voice segments grouped by voice_cluster_id.

        Args:
            limit: Maximum number of segments to retrieve.

        Returns:
            Dictionary mapping cluster_id to list of voice segments.
        """
        try:
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            clusters: dict[int, list[dict[str, Any]]] = {}
            for point in resp[0]:
                payload = point.payload or {}
                cluster_id = payload.get("voice_cluster_id", -1)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(
                    {
                        "id": point.id,
                        "media_path": payload.get("media_path"),
                        "start": payload.get("start"),
                        "end": payload.get("end"),
                        "speaker_label": payload.get("speaker_label"),
                        "speaker_name": payload.get("speaker_name"),
                        "audio_path": payload.get("audio_path"),
                        "voice_cluster_id": cluster_id,
                    }
                )
            return clusters
        except Exception:
            return {}

    def merge_voice_clusters(self, from_cluster: int, to_cluster: int) -> int:
        """Merge two voice clusters into one.

        Args:
            from_cluster: The cluster ID to merge from.
            to_cluster: The cluster ID to merge into.

        Returns:
            Number of segments updated.
        """
        try:
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="voice_cluster_id",
                            match=models.MatchValue(value=from_cluster),
                        )
                    ]
                ),
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            updated = 0
            for point in resp[0]:
                self.client.set_payload(
                    collection_name=VOICE_COLLECTION,
                    payload={"voice_cluster_id": to_cluster},
                    points=[point.id],
                )
                updated += 1
            return updated
        except Exception:
            return 0

    def delete_voice_cluster(self, cluster_id: int | ClusterId) -> int:
        """Delete an entire voice cluster and all its segments.

        Args:
            cluster_id: The voice cluster ID to delete.

        Returns:
            Number of segments deleted.
        """
        try:
            # 1. Get all segments in this cluster
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="voice_cluster_id",
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

            # 2. Delete audio files
            for point in points:
                payload = point.payload or {}
                audio_path = payload.get("audio_path")
                if audio_path:
                    try:
                        if audio_path.startswith("/"):
                            file_path = settings.cache_dir / audio_path.lstrip(
                                "/"
                            )
                            if file_path.exists():
                                file_path.unlink()
                    except Exception:
                        pass

            # 3. Delete the points
            point_ids = [point.id for point in points]
            self.client.delete(
                collection_name=VOICE_COLLECTION,
                points_selector=models.PointIdsList(points=point_ids),
            )
            log(
                f"[DB] Deleted voice cluster {cluster_id}: {len(point_ids)} segments"
            )
            return len(point_ids)
        except Exception as e:
            log(f"delete_voice_cluster failed: {e}")
            return 0

    def get_voice_by_audio_path(self, audio_path: str) -> dict[str, Any] | None:
        """Look up a voice segment by its audio_path.

        Args:
            audio_path: The audio clip path stored in the database.

        Returns:
            Voice segment data dict or None if not found.
        """
        try:
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="audio_path",
                            match=models.MatchValue(value=audio_path),
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
                    "start": payload.get("start", 0),
                    "end": payload.get("end", 0),
                    "speaker_label": payload.get("speaker_label"),
                    "speaker_name": payload.get("speaker_name"),
                    "audio_path": payload.get("audio_path"),
                }
            return None
        except Exception as e:
            log(f"get_voice_by_audio_path error: {e}")
            return None

    def get_voice_segments_for_media(
        self, media_path: str
    ) -> list[dict[str, Any]]:
        """Get all voice segments for a specific media file.

        Used for face-audio temporal mapping to find who is speaking
        at a given timestamp.

        Args:
            media_path: Path to the media file.

        Returns:
            List of voice segment dicts with start, end, cluster_id, speaker_name.
        """
        try:
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="media_path",
                            match=models.MatchValue(value=media_path),
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            results = []
            for point in resp[0]:
                payload = point.payload or {}
                results.append(
                    {
                        "id": str(point.id),
                        "media_path": payload.get("media_path"),
                        "start": payload.get("start", 0),
                        "end": payload.get("end", 0),
                        "cluster_id": payload.get("voice_cluster_id"),
                        "speaker_label": payload.get("speaker_label"),
                        "speaker_name": payload.get("speaker_name"),
                        "audio_path": payload.get("audio_path"),
                    }
                )
            return results
        except Exception as e:
            log(f"get_voice_segments_for_media error: {e}")
            return []

    def get_all_voice_segments(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Retrieve all voice segments for listing/management.

        Args:
            limit: Maximum number of segments to return.

        Returns:
            List of voice segments.
        """
        try:
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = []
            for point in resp[0]:
                payload = point.payload or {}
                cluster_id = payload.get("voice_cluster_id", -1)
                results.append(
                    {
                        "id": point.id,
                        "media_path": payload.get("media_path"),
                        "start": payload.get("start"),
                        "end": payload.get("end"),
                        "speaker_label": payload.get("speaker_label"),
                        "speaker_name": payload.get("speaker_name"),
                        "audio_path": payload.get("audio_path"),
                        "voice_cluster_id": cluster_id,
                    }
                )
            return results
        except Exception as e:
            log(f"get_all_voice_segments error: {e}")
            return []

    def get_speaker_cluster_by_name(self, name: str) -> int | None:
        """Find voice cluster ID by HITL-assigned speaker name.

        Used for cross-modal identity linking - when a face is named,
        find if there's a voice cluster with the same name.

        Args:
            name: The speaker name to search for.

        Returns:
            Voice cluster ID if found, None otherwise.
        """
        try:
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="speaker_name",
                            match=models.MatchValue(value=name),
                        )
                    ]
                ),
                limit=1,
                with_payload=["voice_cluster_id"],
            )
            if resp[0]:
                return (resp[0][0].payload or {}).get("voice_cluster_id")
            return None
        except Exception:
            return None

    def link_face_voice_by_name(self, name: str) -> dict[str, Any]:
        """Link face and voice clusters that share the same HITL-assigned name.

        This enables cross-modal search: "Show me when Prakash is speaking"
        will match both face detection (visual) and voice diarization (audio).

        Args:
            name: The shared identity name.

        Returns:
            Dict with face_cluster_id, voice_cluster_id, and linked status.
        """
        face_cluster_id = self.get_face_cluster_by_name(name)
        voice_cluster_id = self.get_speaker_cluster_by_name(name)

        result = {
            "name": name,
            "face_cluster_id": face_cluster_id,
            "voice_cluster_id": voice_cluster_id,
            "linked": face_cluster_id is not None
            and voice_cluster_id is not None,
        }

        if result["linked"]:
            log(
                f"[CrossModal] Linked identity '{name}': face={face_cluster_id}, voice={voice_cluster_id}"
            )

        return result

    def get_speaker_name_by_cluster(
        self, cluster_id: int | ClusterId
    ) -> str | None:
        """Get HITL-assigned name for a speaker cluster.

        Args:
            cluster_id: The speaker cluster ID.

        Returns:
            Name if assigned, None otherwise.
        """
        try:
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="voice_cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=["speaker_name"],
            )
            if resp[0]:
                return (resp[0][0].payload or {}).get("speaker_name")
            return None
        except Exception:
            return None

    def _propagate_speaker_name_to_frames(
        self, cluster_id: int | ClusterId, name: str
    ) -> int:
        """Propagate speaker name to all frames associated with this voice cluster.

        This ensures that when a speaker is named (e.g. "Speaker 1" -> "John"),
        all frames where this speaker is talking become searchable by "John".

        Args:
            cluster_id: The voice cluster ID.
            name: The new name for the speaker.

        Returns:
            Number of updated frames.
        """
        try:
            # 1. Find all segments for this cluster
            segments = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="voice_cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=1000,
                with_payload=["media_path", "start", "end"],
            )[0]

            if not segments:
                return 0

            updated_frames = 0

            # Group by media path to minimize DB queries
            media_segments = {}
            for seg in segments:
                payload = seg.payload or {}
                path = payload.get("media_path")
                if path:
                    if path not in media_segments:
                        media_segments[path] = []
                    media_segments[path].append(
                        (payload.get("start", 0), payload.get("end", 0))
                    )

            # 2. Update frames for each media file
            for media_path, time_ranges in media_segments.items():
                # Find frames within these time ranges
                # This is an approximation; ideally we'd use exact timestamp matching,
                # but for search metadata, coarse matching is sufficient.

                # Fetch all frames for this video
                frames_resp = self.client.scroll(
                    collection_name=MEDIA_COLLECTION,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="video_path",
                                match=models.MatchValue(value=media_path),
                            )
                        ]
                    ),
                    limit=2000,  # Assume max 2000 frames per video for now or iterate
                    with_payload=True,
                )[0]

                for frame in frames_resp:
                    ts = (frame.payload or {}).get("timestamp", 0)
                    # Check if frame timestamp falls in any speaker segment
                    if any(start <= ts <= end for start, end in time_ranges):
                        # Update this frame
                        frame_id = str(frame.id)
                        payload = frame.payload or {}
                        face_names = payload.get("face_names", [])
                        speaker_names = list(
                            {*payload.get("speaker_names", []), name}
                        )

                        if self.update_frame_identity_text(
                            frame_id, face_names, speaker_names
                        ):
                            updated_frames += 1

            log(
                f"[HITL] Propagated speaker '{name}' to {updated_frames} frames"
            )
            return updated_frames

        except Exception as e:
            log(f"Failed to propagate speaker name: {e}")
            return 0

    def re_embed_voice_cluster_frames(
        self,
        cluster_id: int | ClusterId,
        new_name: str,
        old_name: str | None = None,
    ) -> int:
        """Update and re-embed all frames associated with a voice cluster.

        Args:
            cluster_id: The voice cluster ID being renamed.
            new_name: The new speaker name.
            old_name: The previous speaker name (to remove from lists).

        Returns:
            Number of frames updated.
        """
        try:
            # 1. Get all voice segments for this cluster
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="voice_cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=10000,  # Assume reasonable limit for single cluster
                with_payload=True,
            )
            segments = resp[0]

            if not segments:
                return 0

            updated_count = 0
            processed_frames = set()  # Avoid double processing same frame

            # 2. Iterate segments and find frames
            for seg in segments:
                payload = seg.payload or {}
                media_path = payload.get("media_path") or payload.get(
                    "audio_path"
                )
                start = payload.get("start", 0)
                end = payload.get("end", 0)

                if not media_path:
                    continue

                # Find frames in this time range for this video
                frames_resp = self.client.scroll(
                    collection_name=MEDIA_COLLECTION,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="media_path",
                                match=models.MatchValue(value=media_path),
                            ),
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(gte=start, lte=end),
                            ),
                        ]
                    ),
                    with_payload=True,
                    limit=100,  # usually a few frames per segment
                )
                frames = frames_resp[0]

                for frame in frames:
                    if frame.id in processed_frames:
                        continue

                    current_payload = frame.payload or {}
                    face_names = current_payload.get("face_names", [])
                    speaker_names = current_payload.get("speaker_names", [])

                    # Update speaker names list
                    # Remove old name if it exists
                    if old_name and old_name in speaker_names:
                        speaker_names = [
                            n for n in speaker_names if n != old_name
                        ]

                    # Add new name if not present
                    if new_name not in speaker_names:
                        speaker_names.append(new_name)

                    # 3. Call update_frame_identity_text to re-embed
                    if self.update_frame_identity_text(
                        str(frame.id), face_names, speaker_names
                    ):
                        updated_count += 1

                    processed_frames.add(frame.id)

            return updated_count

        except Exception as e:
            log(f"Error re-embedding voice cluster frames: {e}")
            return 0

    def get_unresolved_voices(self, limit: int = 100) -> list[dict]:
        """Get voice segments that are part of unnamed clusters.

        Returns:
            List of voice segment dictionaries with flat structure.
        """
        try:
            resp = self.client.scroll(
                collection_name=VOICE_COLLECTION,
                scroll_filter=models.Filter(
                    should=[
                        models.IsNullCondition(
                            is_null=models.PayloadField(key="name")
                        ),
                        models.FieldCondition(
                            key="name", match=models.MatchValue(value="")
                        ),
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            unresolved = []
            for p in resp[0]:
                payload = p.payload or {}
                name = payload.get("name")
                if not name:
                    cluster_id = payload.get("cluster_id")
                    if cluster_id is None:
                        cluster_id = abs(hash(str(p.id))) % (10**9)
                    unresolved.append(
                        {
                            "id": str(p.id),
                            "cluster_id": cluster_id,
                            "name": name,
                            "media_path": payload.get("media_path"),
                            "audio_path": payload.get("audio_path"),
                            "start_time": payload.get("start_time"),
                            "end_time": payload.get("end_time"),
                            "duration": payload.get("duration"),
                            "is_main": payload.get("is_main", False),
                        }
                    )
            return unresolved
            return unresolved
        except Exception as e:
            log(f"get_unresolved_voices failed: {e}")
            return []

    async def get_voice_segments_in_range(
        self,
        media_path: str | VideoPath,
        start_time: float | Timestamp,
        end_time: float | Timestamp,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get voice segments overlapping a time range for a specific video.

        Used by agentic_search for voice enrichment of search results.
        """
        try:
            all_segments = self.get_voice_segments_by_video(media_path)
            filtered = []
            for seg in all_segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                # Include if any overlap with the requested range
                if seg_start <= end_time and seg_end >= start_time:
                    filtered.append(seg)
                    if len(filtered) >= limit:
                        break
            return filtered
        except Exception as e:
            log(f"get_voice_segments_in_range failed: {e}")
            return []
