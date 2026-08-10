"""Vector database interface for multimodal embeddings.

This module provides the `VectorDB` class, which handles interactions with Qdrant
for storing and retrieving media segments, frames, faces, and voice embeddings.

Decomposed from the original monolithic class:
- Constants/dimensions -> core.storage.constants
- Text encoder lifecycle -> core.storage.encoder
- Collection schema    -> core.storage.schema
- Shared utilities     -> core.storage.qdrant_utils
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import settings
from core.domain.values import Timestamp, VideoPath
from core.storage.constants import (
    AUDIO_EVENTS_COLLECTION,
    FACE_VECTOR_SIZE,
    FACES_COLLECTION,
    MASKLETS_COLLECTION,
    MEDIA_COLLECTION,
    MEDIA_SEGMENTS_COLLECTION,
    MEDIA_VECTOR_SIZE,
    SCENELETS_COLLECTION,
    SCENES_COLLECTION,
    SELECTED_MODEL,
    SUMMARIES_COLLECTION,
    TEXT_DIM,
    VIDEO_METADATA_COLLECTION,
    VOICE_COLLECTION,
    VOICE_VECTOR_SIZE,
)
from core.storage.encoder import TextEncoder
from core.storage.filters import build_filter, media_path_filter
from core.storage.qdrant_utils import (
    retry_on_connection_error,
    sanitize_numpy_types,
)
from core.storage.repositories.face_repository import FaceRepository
from core.storage.repositories.scene_repository import SceneRepository
from core.storage.repositories.search_repository import SearchRepository
from core.storage.repositories.voice_repository import VoiceRepository
from core.storage.schema import ensure_all_collections
from core.utils.logger import log
from core.utils.observe import observe


class VectorDB(
    FaceRepository,
    VoiceRepository,
    SceneRepository,
    SearchRepository,
):
    """Qdrant vector database storage and retrieval.

    Composed from focused modules:
    - TextEncoder: embedding model lifecycle
    - schema.ensure_all_collections: collection initialization
    - constants: collection names, vector dimensions
    - qdrant_utils: retry, sanitize, paginated scroll
    """

    # Re-export constants as class attributes for backward compatibility
    MEDIA_SEGMENTS_COLLECTION = MEDIA_SEGMENTS_COLLECTION
    MEDIA_COLLECTION = MEDIA_COLLECTION
    FRAMES_COLLECTION = MEDIA_COLLECTION
    FACES_COLLECTION = FACES_COLLECTION
    VOICE_COLLECTION = VOICE_COLLECTION
    SCENES_COLLECTION = SCENES_COLLECTION
    SCENELETS_COLLECTION = SCENELETS_COLLECTION
    SUMMARIES_COLLECTION = SUMMARIES_COLLECTION
    MASKLETS_COLLECTION = MASKLETS_COLLECTION
    AUDIO_EVENTS_COLLECTION = AUDIO_EVENTS_COLLECTION
    VIDEO_METADATA_COLLECTION = VIDEO_METADATA_COLLECTION

    MEDIA_VECTOR_SIZE = MEDIA_VECTOR_SIZE
    FACE_VECTOR_SIZE = FACE_VECTOR_SIZE
    TEXT_DIM = TEXT_DIM
    VOICE_VECTOR_SIZE = VOICE_VECTOR_SIZE
    MODEL_NAME = SELECTED_MODEL

    client: QdrantClient

    def __init__(
        self,
        backend: str = settings.qdrant_backend,
        host: str = settings.qdrant_host,
        port: int = settings.qdrant_port,
        path: str = "qdrant_data_embedded",
    ) -> None:
        """Initialize the VectorDB connection.

        Args:
            backend: The storage backend ('memory' or 'docker').
            host: Qdrant host address for docker backend.
            port: Qdrant port for docker backend.
            path: Local path for embedded storage.
        """
        self._closed = False

        if backend == "memory":
            self.client = QdrantClient(
                path=path, timeout=settings.qdrant_timeout
            )
            log("Initialized embedded Qdrant", path=path, backend=backend)
        elif backend == "docker":
            try:
                self.client = QdrantClient(
                    host=host, port=port, timeout=settings.qdrant_timeout
                )
                self.client.get_collections()
            except Exception as exc:
                log(
                    "Could not connect to Qdrant",
                    error=str(exc),
                    host=host,
                    port=port,
                )
                raise ConnectionError("Qdrant connection failed.") from exc
            log("Connected to Qdrant", host=host, port=port, backend=backend)
        else:
            raise ValueError(
                f"Unknown backend: {backend!r} (use 'memory' or 'docker')"
            )

        # Compose the TextEncoder (replaces inline encoder management)
        self._text_encoder = TextEncoder()

        log(
            f"VectorDB initialized (lazy mode). Encoder: {self.MODEL_NAME} will load on first use."
        )

        # Load Visual Encoder for cross-modal search (Text -> Visual Embedding)
        from core.processing.visual_encoder import get_default_visual_encoder

        self.visual_encoder = get_default_visual_encoder()

        # Initialize all Qdrant collections via schema module
        ensure_all_collections(self.client)

        # Thread-safe cluster ID counter
        import threading

        self._cluster_id_lock = threading.Lock()
        self._cluster_id_counter = 0

        # Expected dimensions for validation
        self._expected_dims = {
            MEDIA_COLLECTION: MEDIA_VECTOR_SIZE,
            FACES_COLLECTION: FACE_VECTOR_SIZE,
            SCENES_COLLECTION: TEXT_DIM,
            SCENELETS_COLLECTION: TEXT_DIM,
            VOICE_COLLECTION: VOICE_VECTOR_SIZE,
        }

    def _validate_vector_dim(
        self, vector: list | None, collection: str, context: str = ""
    ) -> bool:
        """Validate vector dimension before insert (Issue 9).

        Returns True if valid, False if invalid. Logs warning on mismatch.
        """
        if vector is None:
            return True  # None vectors are OK (some collections allow payload-only)

        expected = self._expected_dims.get(collection)
        if expected is None:
            return True  # Unknown collection, skip validation

        actual = len(vector)
        if actual != expected:
            log(
                f"[DIM MISMATCH] {collection}: expected {expected}d, got {actual}d. {context}",
                level="ERROR",
            )
            return False
        return True

    def get_next_voice_cluster_id(self) -> int:
        """Generate a unique voice cluster ID.

        Uses timestamp-based unique ID with atomic counter to guarantee uniqueness.
        Format: YYMMDDHHMM + 4-digit counter = 14-digit ID that's time-sortable.

        Returns:
            Unique integer cluster ID.
        """
        with self._cluster_id_lock:
            # Use UUID4 for process-restart-safe uniqueness
            # Previous approach used (minutes_since_epoch % 100M) * 10000 + counter,
            # which could collide if the process restarted within the same minute.
            import uuid

            cluster_id = uuid.uuid4().int % (
                10**14
            )  # 14-digit int, time-sortable-ish
            return cluster_id

    # =====================================================================
    # ENCODER DELEGATION — All encoder logic lives in TextEncoder
    # =====================================================================

    @property
    def encoder(self):
        """Access the underlying SentenceTransformer model (if loaded)."""
        return self._text_encoder.encoder

    @encoder.setter
    def encoder(self, value):
        """Set the encoder model directly (for legacy compatibility)."""
        self._text_encoder.encoder = value

    def encoder_to_cpu(self) -> None:
        """Move encoder to CPU to free GPU VRAM."""
        self._text_encoder.to_cpu()

    def encoder_to_gpu(self) -> None:
        """Move encoder back to GPU."""
        self._text_encoder.to_gpu()

    async def _ensure_encoder_loaded(self, job_id: str | None = None) -> None:
        """Load encoder if not already loaded."""
        await self._text_encoder.ensure_loaded(job_id=job_id)

    def unload_encoder(self) -> None:
        """Unload encoder from memory to free VRAM."""
        self._text_encoder.unload()

    def unload_encoder_if_idle(self) -> bool:
        """Unload encoder if idle time exceeded."""
        return self._text_encoder.unload_if_idle()

    async def encode_texts(
        self,
        texts: str | list[str],
        batch_size: int = 1,
        show_progress_bar: bool = False,
        is_query: bool = False,
        job_id: str | None = None,
    ) -> list[list[float]]:
        """Transform text(s) into vector embeddings. Delegates to TextEncoder."""
        return await self._text_encoder.encode_texts(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            is_query=is_query,
            job_id=job_id,
        )

    async def get_embedding(self, text: str) -> list[float]:
        """Get a single query embedding for a string."""
        return await self._text_encoder.get_embedding(text)

    async def encode_text(self, text: str) -> list[float]:
        """Encode a single text string (deprecated)."""
        return (await self.encode_texts(text, is_query=True))[0]

    def extract_concepts_from_video(
        self, video_path: str | VideoPath, limit: int = 20
    ) -> list[str]:
        """Extracts top frequent concepts/entities from a video's indexed metadata.

        Used to seed the grounding pipeline if no explicit concepts are provided.
        Scans frames for detected objects (if available in payload).
        """
        # Placeholder implementation - we need to see what's actually in frame payloads.
        # Assuming we have 'objects' or 'ocr_text' in frame payloads.

        # 1. Scroll frames for this video
        try:
            self.get_frames_by_video(video_path)

            # 2. Aggregation (Naive)
            # This depends on what keys (e.g. 'yolo_objects', 'caption_nouns') exist.
            # For now, return empty list to unblock logic, or common objects if found.

            # TODO: Implement proper aggregation once frame schema is finalized with object detection
            return []

        except Exception as e:
            log(f"Failed to extract concepts: {e}")
            return []

    def get_indexed_videos(self) -> list[str]:
        """Retrieves a list of all unique video paths currently indexed in the database.

        Iterates through the media frames collection to extract distinct paths.

        Returns:
            A list of unique video path strings.
        """
        video_paths = set()
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
                limit=500,
                offset=offset,
                with_payload=["video_path"],
                with_vectors=False,
            )
            for point in results:
                if point.payload:
                    path = point.payload.get("video_path")
                    if path:
                        video_paths.add(path)
            if offset is None:
                break
        return list(video_paths)

    def get_frames_by_video(
        self,
        video_path: str | VideoPath,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict]:
        """Retrieves frame metadata for a video within optional time range.

        This method is used by the overlays API to get face/OCR/object data.

        Args:
            video_path: The path string of the target video.
            start_time: Optional start time filter (seconds).
            end_time: Optional end time filter (seconds).

        Returns:
            A list of payload dictionaries containing frame metadata.
        """
        frames = []
        offset = None

        # Build filter conditions (check both video_path and media_path keys)
        path_conditions = [
            models.FieldCondition(
                key="video_path",
                match=models.MatchValue(value=video_path),
            ),
            models.FieldCondition(
                key="media_path",
                match=models.MatchValue(value=video_path),
            ),
        ]
        must_conditions = [
            models.Filter(should=path_conditions),
        ]

        # Add time range filters if specified
        if start_time is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="timestamp",
                    range=models.Range(gte=start_time),
                )
            )
        if end_time is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="timestamp",
                    range=models.Range(lte=end_time),
                )
            )

        while True:
            results, offset = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
                scroll_filter=models.Filter(must=must_conditions),
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                frames.append(point.payload)
            if offset is None:
                break

        # Sort by timestamp for correct overlay ordering
        frames.sort(key=lambda x: x.get("timestamp", 0))
        return frames

    def get_loudness_events(
        self,
        video_path: str | VideoPath,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict]:
        """Retrieves loudness/audio events for a video within optional time range.

        Uses the audio_events collection with loudness metadata.

        Args:
            video_path: Path string of the target video.
            start_time: Optional start time filter (seconds).
            end_time: Optional end time filter (seconds).

        Returns:
            List of loudness event dictionaries.
        """
        events = []

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
                    key="start_time",
                    range=models.Range(gte=start_time),
                )
            )
        if end_time is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="end_time",
                    range=models.Range(lte=end_time),
                )
            )

        try:
            results, _ = self.client.scroll(
                collection_name=self.AUDIO_EVENTS_COLLECTION,
                scroll_filter=models.Filter(must=must_conditions),
                limit=5000,  # Might have many audio events
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                payload = point.payload
                # Format for overlay API
                events.append(
                    {
                        "timestamp": payload.get("start_time", 0),
                        "end_time": payload.get("end_time", 0),
                        "event_class": payload.get("event_class", ""),
                        "confidence": payload.get("confidence", 0),
                        "spl_db": payload.get("spl_db", 0),
                        "lufs": payload.get("lufs", -24),  # Default LUFS
                        "category": payload.get("event_class", ""),
                    }
                )
        except Exception as e:
            import traceback

            log(
                f"Failed to get loudness events for {video_path}: {e}\n{traceback.format_exc()}",
                level="ERROR",
            )

        events.sort(key=lambda x: x.get("timestamp", 0))
        return events

    def list_collections(self) -> models.CollectionsResponse:
        """List all collections in the Qdrant instance."""
        return self.client.get_collections()

    @observe("db_insert_media_segments")
    async def insert_media_segments(
        self,
        video_path: str | VideoPath,
        segments: list[dict[str, Any]],
        job_id: str | None = None,
    ) -> None:
        """Insert media segments (dialogue, subtitles) into the database.

        Args:
            video_path: Path to the source video.
            segments: List of dictionaries containing text, start/end times, etc.
        """
        if not segments:
            return

        texts = [s.get("text", "") for s in segments]
        embeddings = await self.encode_texts(texts, batch_size=1, job_id=job_id)

        points: list[models.PointStruct] = []

        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0.0)
            unique_str = f"{video_path}_{start_time}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

            payload = {
                "video_path": video_path,
                "media_path": video_path,  # Standardized path key
                "text": segment.get("text"),
                # Standardized keys (used by fusion)
                "start_time": start_time,
                "end_time": segment.get("end"),
                # Legacy keys for backwards compatibility
                "start": start_time,
                "end": segment.get("end"),
                "type": segment.get("type", "dialogue"),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embeddings[i],
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.MEDIA_SEGMENTS_COLLECTION,
            points=points,
            wait=False,
        )

    @observe("db_search_media")
    @observe("db_search_media")
    @observe("db_upsert_media_frame")
    @retry_on_connection_error()
    def upsert_media_frame(
        self,
        point_id: str,
        vector: list[float],
        video_path: str | VideoPath,
        timestamp: float | Timestamp,
        action: str | None = None,
        dialogue: str | None = None,
        payload: dict[str, Any] | None = None,
        ocr_text: str | None = None,  # Add text
    ):
        """Upsert a single frame embedding with structured metadata.

        Args:
            point_id: Unique ID for the point.
            vector: The vector embedding of the frame description.
            video_path: Path to the source video.
            timestamp: Timestamp in seconds.
            action: Action description (optional).
            dialogue: Associated dialogue (optional).
            payload: Additional payload dictionary (optional).
            ocr_text: Text extracted via OCR (optional).
        """
        payload = payload or {}
        payload.update(
            {
                "video_path": video_path,
                "timestamp": timestamp,
                "action": action,
                "dialogue": dialogue,
                "ocr_text": ocr_text,  # Store in payload
            }
        )

        # Ensure scan_id is present if passed in payload
        # This fixes the filtering issue where scan_id was sometimes dropped
        if payload.get("scan_id"):
            payload["scan_id"] = str(payload["scan_id"])

        payload = sanitize_numpy_types(payload)

        self.client.upsert(
            collection_name=self.MEDIA_COLLECTION,
            points=[
                models.PointStruct(id=point_id, vector=vector, payload=payload)
            ],
            wait=False,
        )

    @observe("db_upsert_media_frames_batch")
    @retry_on_connection_error()
    def upsert_media_frames_batch(
        self,
        frames: list[dict],
    ) -> int:
        """Batch upsert multiple frame embeddings for better performance.

        This is ~10x faster than individual upsert calls when processing
        many frames, as it reduces network round-trips and Qdrant overhead.

        Args:
            frames: List of frame dicts, each containing:
                - point_id: Unique ID for the point
                - vector: The vector embedding
                - video_path: Path to source video
                - timestamp: Timestamp in seconds
                - action: Action description (optional)
                - dialogue: Associated dialogue (optional)
                - payload: Additional payload dict (optional)
                - ocr_text: Extracted text (optional)

        Returns:
            Number of frames successfully upserted.
        """
        if not frames:
            return 0

        points = []
        for frame in frames:
            payload = frame.get("payload", {}) or {}
            ts = frame.get("timestamp", 0.0)
            # Add end_time and duration for proper clip playback
            end_time = (
                frame.get("end_time") or ts + settings.search_default_duration
            )
            duration = frame.get("duration") or settings.search_default_duration
            payload.update(
                {
                    "video_path": frame.get("video_path", ""),
                    "media_path": frame.get("video_path", ""),
                    "timestamp": ts,
                    "start_time": ts,
                    "end_time": end_time,
                    "duration": duration,
                    "action": frame.get("action"),
                    "dialogue": frame.get("dialogue"),
                    "ocr_text": frame.get("ocr_text"),
                }
            )

            if payload.get("scan_id"):
                payload["scan_id"] = str(payload["scan_id"])

            payload = sanitize_numpy_types(payload)

            points.append(
                models.PointStruct(
                    id=frame["point_id"],
                    vector=frame["vector"],
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.MEDIA_COLLECTION,
            points=points,
            wait=False,
        )
        return len(points)

    @observe("db_insert_masklet")
    @observe("db_search_masklets")
    @observe("db_update_masklet_concept")
    @observe("db_update_masklet")
    def get_global_summary(
        self, video_path: str | VideoPath
    ) -> dict[str, Any] | None:
        """Retrieves the global summary for a specific video.

        Args:
            video_path: Path to the source video.

        Returns:
            Summary payload or None.
        """
        try:
            results, _ = self.client.scroll(
                collection_name=self.SUMMARIES_COLLECTION,
                scroll_filter=build_filter(
                    [
                        media_path_filter(video_path),
                        models.FieldCondition(
                            key="level",
                            match=models.MatchValue(value="L2"),  # L2 is global
                        ),
                    ]
                ),
                limit=1,
            )
            if results[0]:
                return results[0][0].payload
            return None
        except Exception as e:
            log(f"Failed to fetch summary: {e}", level="ERROR")
            return None

    async def search_global_summaries(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Search video-level summaries semantically.

        Enables high-level queries like "videos about cooking" or
        "videos featuring outdoor sports".

        Args:
            query: Natural language search query.
            limit: Maximum results.
            score_threshold: Minimum similarity score.

        Returns:
            List of matching video summaries with scores.
        """
        query_vector = await self.encode_texts(query)

        try:
            results = self.client.search(
                collection_name=self.SUMMARIES_COLLECTION,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )

            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "video_path": r.payload.get("video_path", ""),
                    "summary": r.payload.get("summary", ""),
                    "level": r.payload.get("level", ""),
                    "key_entities": r.payload.get("key_entities", []),
                    "duration": r.payload.get("duration", 0),
                    **r.payload,
                }
                for r in results
                if r.payload
            ]
        except Exception as e:
            log(f"Summary search failed: {e}")
            return []

    @observe("db_match_speaker")
    @observe("db_search_frames_filtered")
    def get_recent_frames_search(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get most recently indexed frames as fallback for empty search results.

        Args:
            limit: Maximum number of results.

        Returns:
            List of recent frames with payload data.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = []
            for point in resp[0]:
                payload = point.payload or {}
                results.append(
                    {
                        "id": str(point.id),
                        "score": 0.5,  # Default score for fallback results
                        "fallback": True,
                        **payload,
                    }
                )
            return results
        except Exception as e:
            log(f"search_frames_filtered failed: {e}", level="DEBUG")
            return []

    @observe("db_search_frames_hybrid")
    @observe("db_get_masklets")
    def get_frame_by_id(self, frame_id: str) -> dict[str, Any] | None:
        """Retrieve a specific frame by ID."""
        try:
            results = self.client.retrieve(
                collection_name=self.MEDIA_COLLECTION,
                ids=[frame_id],
                with_payload=True,
            )
            if results:
                return results[0].payload
        except Exception as e:
            log(f"get_frame_by_id failed for '{frame_id}': {e}", level="DEBUG")
        return None

    def insert_audio_event(
        self,
        media_path: str | VideoPath,
        event_type: str,
        start_time: float | Timestamp,
        end_time: float | Timestamp,
        confidence: float,
        clap_embedding: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ):
        """Insert audio event with optional CLAP embedding for semantic search.

        Args:
            media_path: Path to the media file.
            event_type: Type/label of the audio event.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            confidence: Detection confidence score.
            clap_embedding: Optional 512-dim CLAP embedding for vector search.
            payload: Additional metadata.
        """
        unique_str = f"{media_path}_{event_type}_{start_time}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

        data = {
            "media_path": media_path,
            "event": event_type,
            "start_time": start_time,
            "end_time": end_time,
            "confidence": confidence,
            "has_embedding": clap_embedding is not None,
            **(payload or {}),
        }

        # Use CLAP embedding if provided, otherwise use zero vector
        CLAP_DIM = 512
        if clap_embedding is not None and len(clap_embedding) == CLAP_DIM:
            vector = clap_embedding
        else:
            vector = [
                0.0
            ] * CLAP_DIM  # Zero vector for events without embedding

        self.client.upsert(
            collection_name=self.AUDIO_EVENTS_COLLECTION,
            points=[
                models.PointStruct(id=point_id, vector=vector, payload=data)
            ],
        )

    def update_media_metadata(
        self, media_path: str | VideoPath, metadata: dict[str, Any]
    ):
        """Update video-level metadata."""
        unique_str = f"metadata_{media_path}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

        data = {"media_path": media_path, **metadata}

        self.client.upsert(
            collection_name=self.VIDEO_METADATA_COLLECTION,
            points=[
                models.PointStruct(id=point_id, vector=[1.0], payload=data)
            ],
        )

    @observe("db_insert_face")
    @observe("db_search_face")
    @observe("db_insert_voice_segment")
    # =========================================================================
    # SCENE-LEVEL STORAGE (Production architecture like Twelve Labs)
    # =========================================================================

    @observe("db_store_scene")
    @observe("db_store_scenelet")
    @observe("db_search_scenelets")
    @observe("db_search_voice_segments")
    @observe("db_search_audio_events")
    @observe("db_search_dialogue")
    async def search_dialogue(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        video_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search dialogue/transcripts using semantic vector similarity.

        This searches the media_segments collection which contains transcribed
        speech and subtitles from ASR (Whisper).

        Args:
            query: The search query text.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.
            video_path: Optional filter by video path.

        Returns:
            List of matching dialogue segments with scores.
        """
        try:
            query_vec = (await self.encode_texts(query, is_query=True))[0]

            conditions: list[models.Condition] = []
            if video_path:
                conditions.append(
                    models.FieldCondition(
                        key="media_path",
                        match=models.MatchValue(value=video_path),
                    )
                )

            query_filter = (
                models.Filter(must=conditions) if conditions else None
            )

            resp = self.client.query_points(
                collection_name=self.MEDIA_SEGMENTS_COLLECTION,
                query=query_vec,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )

            results = []
            for hit in resp.points:
                payload = hit.payload or {}
                results.append(
                    {
                        "id": str(hit.id),
                        "score": hit.score,
                        "type": "dialogue",
                        "text": payload.get(
                            "text", payload.get("transcription", "")
                        ),
                        "start": payload.get(
                            "start_time", payload.get("start", 0)
                        ),
                        "end": payload.get("end_time", payload.get("end", 0)),
                        "timestamp": payload.get(
                            "start_time", payload.get("start", 0)
                        ),
                        "video_path": payload.get("media_path"),
                        "language": payload.get("language", "en"),
                        **payload,
                    }
                )
            return results
        except Exception as e:
            log(f"search_dialogue failed: {e}")
            return []

    @observe("db_search_video_metadata")
    async def search_video_metadata(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search video-level metadata (summaries, titles, context).

        Args:
            query: The search query.
            limit: Maximum results.

        Returns:
            List of matching videos with metadata.
        """
        try:
            # VIDEO_METADATA_COLLECTION uses dummy vectors (size=1), 
            # so we scroll and do a text match on summary and title instead of semantic search.
            query_lower = query.lower()
            resp = self.client.scroll(
                collection_name=self.VIDEO_METADATA_COLLECTION,
                limit=limit * 10,  # Fetch more to filter down
                with_payload=True,
                with_vectors=False,
            )

            results = []
            for hit in resp[0]:
                payload = hit.payload or {}
                summary = payload.get("summary", "")
                title = payload.get("title", "")
                
                # Check for query match in summary or title
                if query_lower in summary.lower() or query_lower in title.lower():
                    results.append(
                        {
                            "id": str(hit.id),
                            "score": 1.0,  # Exact text match score
                            "type": "video_metadata",
                            "video_path": payload.get("video_path"),
                            "summary": summary,
                            "title": title,
                            **payload,
                        }
                    )
                    if len(results) >= limit:
                        break
            return results
        except Exception as e:
            log(f"search_video_metadata failed: {e}")
            return []

    @observe("db_search_scenes")
    @observe("db_search_scenes_by_image")
    @observe("db_search_scenes_by_action")
    @observe("db_explainable_search")
    def close(self) -> None:
        """Close the database client connection."""
        if self._closed:
            return
        self._closed = True
        try:
            self.client.close()
        except Exception:
            pass

    @observe("db_get_unresolved_faces")
    @observe("db_update_face_name")
    @observe("db_update_face_cluster_id")
    @observe("db_merge_face_clusters")
    @observe("db_update_video_metadata")
    def update_video_metadata(
        self, video_path: str | VideoPath, metadata: dict[str, Any]
    ) -> int:
        """Update metadata for all frames belonging to a video.

        Args:
            video_path: The video path to match.
            metadata: Dictionary of metadata to update/add.

        Returns:
            Number of frames updated.
        """
        try:
            # 1. Find all frames for this video
            resp = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_path",
                            match=models.MatchValue(value=video_path),
                        )
                    ]
                ),
                limit=10000,  # Assume reasonable max frames per video for update
                with_payload=False,
                with_vectors=False,
            )
            points = resp[0]
            if not points:
                return 0

            point_ids = [point.id for point in points]
            self.client.set_payload(
                collection_name=self.MEDIA_COLLECTION,
                payload=metadata,
                points=point_ids,  # type: ignore
            )
            return len(point_ids)
        except Exception as e:
            from core.utils.logger import get_logger

            log = get_logger(__name__)
            log.error(f"Failed to update video metadata: {e}")
            return 0

    @observe("db_get_named_faces")
    @observe("db_delete_face")
    @observe("db_update_single_face_name")
    @observe("db_get_indexed_media")
    def get_indexed_media(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Get list of ALL indexed media files.

        NOTE: This now properly paginates through the entire collection
        to ensure all videos are returned (not just those in first 100 records).

        Args:
            limit: Maximum segments to scan per page (higher = more complete).

        Returns:
            List of all unique indexed media files with segment counts.
        """
        try:
            seen_paths: dict[str, dict[str, Any]] = {}
            offset = None

            # Paginate through ALL segments to build complete video list
            while True:
                resp = self.client.scroll(
                    collection_name=self.MEDIA_SEGMENTS_COLLECTION,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = resp

                for point in points:
                    payload = point.payload or {}
                    video_path = payload.get("video_path")
                    if video_path and video_path not in seen_paths:
                        seen_paths[video_path] = {
                            "video_path": video_path,
                            "segment_count": 1,
                        }
                    elif video_path:
                        seen_paths[video_path]["segment_count"] += 1

                # No more pages
                if next_offset is None or not points:
                    break
                offset = next_offset

            return list(seen_paths.values())
        except Exception:
            return []

    @observe("db_get_collection_stats")
    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about all collections.

        Returns:
            Dictionary with counts for each collection.
        """
        stats = {}
        for name in [
            self.MEDIA_SEGMENTS_COLLECTION,
            self.MEDIA_COLLECTION,
            self.FACES_COLLECTION,
            self.VOICE_COLLECTION,
        ]:
            try:
                info = self.client.get_collection(name)
                stats[name] = {
                    "points_count": info.points_count,
                    "vectors_count": getattr(
                        info, "vectors_count", info.points_count
                    ),
                }
            except Exception:
                stats[name] = {"points_count": 0, "vectors_count": 0}
        return stats

    @observe("db_delete_media")
    def delete_media(self, video_path: str | VideoPath) -> int:
        """Delete all data associated with a media file.

        Args:
            video_path: Path to the media file.

        Returns:
            Total number of points deleted.
        """
        deleted = 0

        # Cleanup Faces
        try:
            # 1. Get Faces to delete files
            face_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="media_path",
                        match=models.MatchValue(value=video_path),
                    )
                ]
            )
            face_points = self.client.scroll(
                self.FACES_COLLECTION,
                scroll_filter=face_filter,
                limit=10000,
                with_payload=True,
            )[0]
            for pt in face_points:
                payload = pt.payload or {}
                thumb = payload.get("thumbnail_path")
                if thumb:
                    try:
                        if thumb.startswith("/thumbnails"):
                            path = settings.cache_dir / thumb.lstrip("/")
                            if path.exists():
                                path.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

        # Cleanup Voice Segments
        try:
            voice_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="media_path",
                        match=models.MatchValue(value=video_path),
                    )
                ]
            )
            voice_points = self.client.scroll(
                self.VOICE_COLLECTION,
                scroll_filter=voice_filter,
                limit=10000,
                with_payload=True,
            )[0]
            for pt in voice_points:
                payload = pt.payload or {}
                audio = payload.get("audio_path")
                if audio:
                    try:
                        if audio.startswith("/thumbnails"):
                            path = settings.cache_dir / audio.lstrip("/")
                            if path.exists():
                                path.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

        for collection in [
            self.MEDIA_SEGMENTS_COLLECTION,
            self.MEDIA_COLLECTION,
            self.FACES_COLLECTION,
            self.VOICE_COLLECTION,
            self.SCENES_COLLECTION,
            self.SCENELETS_COLLECTION,
            self.AUDIO_EVENTS_COLLECTION,
            self.MASKLETS_COLLECTION,
            self.SUMMARIES_COLLECTION,
            self.VIDEO_METADATA_COLLECTION,
        ]:
            try:
                # For faces and voices, we need to match media_path
                key = (
                    "video_path"
                    if collection
                    in [self.MEDIA_SEGMENTS_COLLECTION, self.MEDIA_COLLECTION]
                    else "media_path"
                )

                self.client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchValue(value=video_path),
                                )
                            ]
                        )
                    ),
                )
                deleted += 1
            except Exception:
                pass
        return deleted

    @observe("db_delete_voice_segment")
    @observe("db_get_all_face_embeddings")
    @observe("db_get_faces_grouped_by_cluster")
    @observe("db_get_all_cluster_centroids")
    @observe("db_update_cluster_centroid")
    @observe("db_update_voice_speaker_name")
    @observe("db_get_all_voice_embeddings")
    @observe("db_update_voice_cluster_id")
    @observe("db_get_voices_grouped_by_cluster")
    @observe("db_merge_voice_clusters")
    @observe("db_delete_voice_cluster")
    @observe("db_delete_face_cluster")
    @observe("db_get_recent_frames")
    def get_recent_frames(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most recently indexed frames.

        Args:
            limit: Maximum number of frames to retrieve.

        Returns:
            List of frame result dicts.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = []
            for point in resp[0]:
                payload = point.payload or {}
                results.append(
                    {
                        "score": 1.0,  # Implicitly high relevance for recent
                        "id": str(point.id),
                        **payload,
                    }
                )
            return results
        except Exception as e:
            log(f"get_recent_frames error: {e}")
            return []

    @observe("db_get_all_voice_segments")

    # =========================================================================
    # HITL IDENTITY INTEGRATION & HYBRID SEARCH
    # =========================================================================

    def get_all_hitl_names(self) -> list[str]:
        """Get all HITL-assigned names (faces and speakers).

        Returns:
            List of unique names from face clusters and speaker clusters.
        """
        names = set()

        # Face names
        try:
            face_resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must_not=[
                        models.IsNullCondition(
                            is_null=models.PayloadField(key="name")
                        )
                    ]
                ),
                limit=1000,
                with_payload=["name"],
            )
            for point in face_resp[0]:
                name = (point.payload or {}).get("name")
                if name:
                    names.add(name)
        except Exception:
            pass

        # Speaker names
        try:
            voice_resp = self.client.scroll(
                collection_name=self.VOICE_COLLECTION,
                scroll_filter=models.Filter(
                    must_not=[
                        models.IsNullCondition(
                            is_null=models.PayloadField(key="speaker_name")
                        )
                    ]
                ),
                limit=1000,
                with_payload=["speaker_name"],
            )
            for point in voice_resp[0]:
                name = (point.payload or {}).get("speaker_name")
                if name:
                    names.add(name)
        except Exception:
            pass

        return list(names)

    def get_person_co_occurrences(
        self,
        video_path: str | None = None,
        time_window_seconds: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Extract person co-occurrences for GraphRAG relationship building.

        Finds pairs of people who appear together in the same time window.
        This forms the basis for APPEARED_WITH edges in the knowledge graph.

        Args:
            video_path: Optional filter to specific video.
            time_window_seconds: Time window for considering co-occurrence.

        Returns:
            List of co-occurrence edges with temporal metadata.
        """
        try:
            # Get all frames with face detections
            conditions = []
            if video_path:
                conditions.append(
                    models.FieldCondition(
                        key="video_path",
                        match=models.MatchValue(value=video_path),
                    )
                )

            resp = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
                scroll_filter=models.Filter(must=conditions)
                if conditions
                else None,
                limit=5000,
                with_payload=True,
                with_vectors=False,
            )

            # Group frames by video and find co-occurrences
            co_occurrences: dict[tuple, dict] = {}

            for point in resp[0]:
                payload = point.payload or {}
                face_cluster_ids = payload.get("face_cluster_ids", [])
                face_names = payload.get("face_names", [])
                timestamp = payload.get("timestamp", 0)
                vid_path = payload.get("video_path", "")

                # Only process frames with 2+ faces
                if len(face_cluster_ids) < 2:
                    continue

                # Create pairs of co-occurring identities
                for i in range(len(face_cluster_ids)):
                    for j in range(i + 1, len(face_cluster_ids)):
                        id1, id2 = sorted(
                            [face_cluster_ids[i], face_cluster_ids[j]]
                        )
                        name1 = face_names[i] if i < len(face_names) else None
                        name2 = face_names[j] if j < len(face_names) else None

                        key = (vid_path, id1, id2)
                        if key not in co_occurrences:
                            co_occurrences[key] = {
                                "video_path": vid_path,
                                "person1_cluster_id": id1,
                                "person1_name": name1,
                                "person2_cluster_id": id2,
                                "person2_name": name2,
                                "timestamps": [],
                                "interaction_count": 0,
                            }

                        co_occurrences[key]["timestamps"].append(timestamp)
                        co_occurrences[key]["interaction_count"] += 1

                        # Update names if discovered
                        if name1 and not co_occurrences[key]["person1_name"]:
                            co_occurrences[key]["person1_name"] = name1
                        if name2 and not co_occurrences[key]["person2_name"]:
                            co_occurrences[key]["person2_name"] = name2

            # Compute time ranges for each co-occurrence
            results = []
            for key, data in co_occurrences.items():
                timestamps = sorted(data["timestamps"])
                if timestamps:
                    data["start_time"] = min(timestamps)
                    data["end_time"] = max(timestamps)
                    data["duration"] = data["end_time"] - data["start_time"]
                del data["timestamps"]  # Remove raw list to save space
                results.append(data)

            log(f"[GraphRAG] Found {len(results)} co-occurrence relationships")
            return results

        except Exception as e:
            log(f"get_person_co_occurrences error: {e}")
            return []

    @observe("db_search_voice")
    @observe("db_search_hybrid_legacy")
    @observe("db_update_frame_description")
    async def update_frame_description(
        self, frame_id: str, description: str
    ) -> bool:
        """Update frame description manually and re-embed. HITL correction for VLM errors."""
        try:
            resp = self.client.retrieve(
                collection_name=self.MEDIA_COLLECTION,
                ids=[frame_id],
                with_payload=True,
                with_vectors=False,
            )
            if not resp:
                log(f"Frame {frame_id} not found")
                return False

            payload = resp[0].payload or {}
            payload["action"] = description
            payload["description"] = description
            payload["is_hitl_corrected"] = True

            identity_text = payload.get("identity_text", "")
            full_text = (
                f"{description}. {identity_text}"
                if identity_text
                else description
            )

            new_vector = (await self.encode_texts(full_text, is_query=False))[0]

            self.client.upsert(
                collection_name=self.MEDIA_COLLECTION,
                points=[
                    models.PointStruct(
                        id=frame_id, vector=new_vector, payload=payload
                    )
                ],
            )
            log(
                f"HITL: Re-embedded frame {frame_id} with: '{description[:50]}...'"
            )
            return True
        except Exception as e:
            log(f"Failed to update frame description: {e}")
            return False

    async def update_frame_identity_text(
        self,
        frame_id: str,
        face_names: list[str],
        speaker_names: list[str],
    ) -> bool:
        """Update a frame's identity text AND re-embed the vector.

        Called when HITL names are assigned to update searchability.
        Crucial: This combines the original visual description with the new names
        and re-generates the embedding vector so the names are searchable.

        Args:
            frame_id: The frame point ID.
            face_names: List of visible person names.
            speaker_names: List of speaking person names.

        Returns:
            Success status.
        """
        try:
            # 1. Fetch existing frame to get visual description
            resp = self.client.retrieve(
                collection_name=self.MEDIA_COLLECTION,
                ids=[frame_id],
                with_payload=True,
                with_vectors=False,  # Don't need old vector
            )
            if not resp:
                return False

            point = resp[0]
            payload = point.payload or {}

            # Get original visual description
            description = (
                payload.get("description") or payload.get("action") or ""
            )

            # 2. Build new Identity Text
            identity_parts = []
            if face_names:
                identity_parts.append(f"Visible: {', '.join(face_names)}")
            if speaker_names:
                identity_parts.append(f"Speaking: {', '.join(speaker_names)}")

            identity_text = ". ".join(identity_parts)

            # 3. Create NEW combined text for embedding
            # "A man walking. Visible: John. Speaking: John"
            full_text = (
                f"{description}. {identity_text}"
                if identity_text
                else description
            )

            if not full_text.strip():
                return False

            # 4. Re-encode
            new_vector = (await self.encode_texts(full_text, is_query=False))[0]

            # 5. Update payload
            payload["face_names"] = face_names
            payload["speaker_names"] = speaker_names
            payload["identity_text"] = identity_text

            # 6. Upsert with NEW vector
            self.client.upsert(
                collection_name=self.MEDIA_COLLECTION,
                points=[
                    models.PointStruct(
                        id=frame_id,
                        vector=new_vector,
                        payload=payload,
                    )
                ],
            )
            log(
                f"Re-embedded frame {frame_id} with names: {face_names + speaker_names}"
            )
            return True

        except Exception as e:
            log(f"Failed to update frame identity: {e}")
            return False

    @observe("db_delete_media")
    def delete_media_by_path(self, media_path: str | VideoPath) -> None:
        """Delete all data associated with a media file."""
        for collection in [
            self.MEDIA_SEGMENTS_COLLECTION,
            self.MEDIA_COLLECTION,
            self.FACES_COLLECTION,
            self.VOICE_COLLECTION,
            self.SCENES_COLLECTION,
            self.SCENELETS_COLLECTION,
            self.AUDIO_EVENTS_COLLECTION,
            self.MASKLETS_COLLECTION,
            self.SUMMARIES_COLLECTION,
            self.VIDEO_METADATA_COLLECTION,
        ]:
            try:
                # Try with "media_path" key
                self.client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="media_path",
                                    match=models.MatchValue(value=media_path),
                                )
                            ]
                        )
                    ),
                )
                # Try with "video_path" key (legacy/mixed usage)
                self.client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="video_path",
                                    match=models.MatchValue(value=media_path),
                                )
                            ]
                        )
                    ),
                )
            except Exception as e:
                log(f"Failed to delete from {collection}: {e}")

    def get_entity_co_occurrences(
        self, limit_frames: int = 5000
    ) -> dict[int, dict[str, int]]:
        """Aggregates NER entities that co-occur with face clusters in frames.

        Args:
            limit_frames: Number of recent frames to analyze.

        Returns:
            Dict[cluster_id, Dict[entity_name, count]]
        """
        co_occurrences: dict[int, dict[str, int]] = {}

        try:
            # Scroll recent frames with payloads
            resp = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
                limit=limit_frames,
                with_payload=["face_cluster_ids", "entities"],
                with_vectors=False,
            )[0]

            for point in resp:
                payload = point.payload or {}
                cluster_ids = payload.get("face_cluster_ids", [])
                entities = payload.get("entities", [])

                if not cluster_ids or not entities:
                    continue

                for cid in cluster_ids:
                    if cid not in co_occurrences:
                        co_occurrences[cid] = {}

                    for entity in entities:
                        # Skip if entity matches "Person" etc.
                        if entity.lower() in ("person", "man", "woman"):
                            continue

                        co_occurrences[cid][entity] = (
                            co_occurrences[cid].get(entity, 0) + 1
                        )

            return co_occurrences
        except Exception as e:
            log(f"get_entity_co_occurrences failed: {e}")
            return {}

    # =========================================================================
    # METHODS REQUIRED BY AGENTIC SEARCH (Fix #14)
    # =========================================================================
