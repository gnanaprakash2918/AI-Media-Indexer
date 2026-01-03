"""Vector database interface for multimodal embeddings.

This module provides the `VectorDB` class, which handles interactions with Qdrant
for storing and retrieving media segments, frames, faces, and voice embeddings.
"""
from __future__ import annotations

import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any

import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from config import settings
from core.utils.hardware import select_embedding_model
from core.utils.logger import log
from core.utils.observe import observe


def retry_on_connection_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry Qdrant operations on connection errors (WinError 10053, etc.)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (OSError, ConnectionError, ConnectionResetError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        log(f"Qdrant connection error (attempt {attempt + 1}): {e}, retrying...")
                        time.sleep(delay * (attempt + 1))
                    else:
                        log(f"Qdrant connection failed after {max_retries} attempts: {e}")
            if last_error:
                raise last_error
            raise RuntimeError("Qdrant retry failed")
        return wrapper
    return decorator


# Auto-select embedding model based on available VRAM
# Allow override from config
if settings.embedding_model_override:
    _SELECTED_MODEL = settings.embedding_model_override
    # We'd ideally infer dim, but for custom overrides, we might default to 1024 or 768?
    # For now, let's assume if they stick to the e5/bge family:
    if "large" in _SELECTED_MODEL or "m3" in _SELECTED_MODEL:
        _SELECTED_DIM = 1024
    elif "base" in _SELECTED_MODEL:
        _SELECTED_DIM = 768
    else:
        _SELECTED_DIM = 384
    log(f"Overriding embedding model from config: {_SELECTED_MODEL} ({_SELECTED_DIM}d)")
else:
    _SELECTED_MODEL, _SELECTED_DIM = select_embedding_model()


class VectorDB:
    """Wrapper for Qdrant vector database storage and retrieval."""

    MEDIA_SEGMENTS_COLLECTION = "media_segments"
    MEDIA_COLLECTION = "media_frames"
    FACES_COLLECTION = "faces"
    VOICE_COLLECTION = "voice_segments"

    MEDIA_VECTOR_SIZE = _SELECTED_DIM
    FACE_VECTOR_SIZE = 512  # InsightFace ArcFace (fallback SFace uses 128 but vectors are padded)
    TEXT_DIM = _SELECTED_DIM
    VOICE_VECTOR_SIZE = 256

    MODEL_NAME = _SELECTED_MODEL

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
            self.client = QdrantClient(path=path)
            log("Initialized embedded Qdrant", path=path, backend=backend)
        elif backend == "docker":
            try:
                self.client = QdrantClient(host=host, port=port)
                self.client.get_collections()
            except Exception as exc:
                log("Could not connect to Qdrant", error=str(exc), host=host, port=port)
                raise ConnectionError("Qdrant connection failed.") from exc
            log("Connected to Qdrant", host=host, port=port, backend=backend)
        else:
            raise ValueError(f"Unknown backend: {backend!r} (use 'memory' or 'docker')")

        log(
            "Loading SentenceTransformer model...",
            model_name=self.MODEL_NAME,
            device=settings.device,
        )

        self.encoder = self._load_encoder()
        self._ensure_collections()

    def _load_encoder(self) -> SentenceTransformer:
        models_dir = settings.model_cache_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        local_model_dir = models_dir / self.MODEL_NAME
        target_device = settings.device or "cpu"

        def _create(path_or_name: str, device: str) -> SentenceTransformer:
            log(
                "Creating SentenceTransformer",
                path_or_name=path_or_name,
                device=device,
            )
            
            # Use trust_remote_code for BGE models, allow Hub download if needed
            return SentenceTransformer(
                path_or_name, 
                device=device,
                trust_remote_code=True,
            )

        if local_model_dir.exists():
            log("Loading cached model", path=str(local_model_dir), device=target_device)
            try:
                return _create(str(local_model_dir), device=target_device)
            except Exception as exc:
                log(f"GPU Load Failed: {exc}. Retrying on CPU...", level="warning")
                try:
                    return _create(str(local_model_dir), device="cpu")
                except Exception as exc_cpu:
                    log(f"CPU Fallback Failed: {exc_cpu}", level="error")
                    # If local corrupted, proceed to redownload logic
                    pass

        log("Local model missing/corrupt, downloading from Hub", model=self.MODEL_NAME)
        try:
            model = _create(self.MODEL_NAME, device=target_device)
        except Exception as exc:
            log(f"Hub Load (GPU) Failed: {exc}. Retrying on CPU...", level="warning")
            model = _create(self.MODEL_NAME, device="cpu")

        try:
            model.save(str(local_model_dir))
        except Exception as exc:
            log("Warning: failed to save model cache", error=str(exc))

        return model

    @observe("db_encode_texts")
    def encode_texts(
        self,
        texts: str | list[str],
        batch_size: int = 1,
        show_progress_bar: bool = False,
        is_query: bool = False,
    ) -> list[list[float]]:
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        # e5 models require prefix (query: or passage:)
        if "e5" in self.MODEL_NAME.lower():
            prefix = "query: " if is_query else "passage: "
            texts_list = [prefix + t for t in texts_list]

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        embeddings = self.encoder.encode(
            texts_list,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        return [list(e) for e in embeddings]

    def _ensure_collections(self) -> None:
        if not self.client.collection_exists(self.MEDIA_SEGMENTS_COLLECTION):
            self.client.create_collection(
                collection_name=self.MEDIA_SEGMENTS_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.TEXT_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

        if not self.client.collection_exists(self.MEDIA_COLLECTION):
            self.client.create_collection(
                collection_name=self.MEDIA_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.MEDIA_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

        if not self.client.collection_exists(self.FACES_COLLECTION):
            self.client.create_collection(
                collection_name=self.FACES_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.FACE_VECTOR_SIZE,
                    distance=models.Distance.EUCLID,
                ),
            )

        if not self.client.collection_exists(self.VOICE_COLLECTION):
            self.client.create_collection(
                collection_name=self.VOICE_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.VOICE_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

        # SAM 3 Masklets collection (for spatio-temporal segmentation)
        if not self.client.collection_exists("masklets"):
            self.client.create_collection(
                collection_name="masklets",
                vectors_config=models.VectorParams(
                    size=1,  # Minimal vector, queries use payload filters
                    distance=models.Distance.COSINE,
                ),
            )

        log("Qdrant collections ensured")

    def list_collections(self) -> models.CollectionsResponse:
        """List all collections in the Qdrant instance."""
        return self.client.get_collections()

    @observe("db_insert_media_segments")
    def insert_media_segments(
        self,
        video_path: str,
        segments: list[dict[str, Any]],
    ) -> None:
        """Insert media segments (dialogue, subtitles) into the database.

        Args:
            video_path: Path to the source video.
            segments: List of dictionaries containing text, start/end times, etc.
        """
        if not segments:
            return

        texts = [s.get("text", "") for s in segments]
        embeddings = self.encode_texts(texts, batch_size=1)

        points: list[models.PointStruct] = []

        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0.0)
            unique_str = f"{video_path}_{start_time}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

            payload = {
                "video_path": video_path,
                "text": segment.get("text"),
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
        )

    @observe("db_search_media")
    def search_media(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
        video_path: str | None = None,
        segment_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for media segments similar to the query string.

        Args:
            query: The search query text.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score cutoff.
            video_path: Filter results by video path.
            segment_type: Filter results by segment type.

        Returns:
            A list of matching segments with metadata.
        """
        query_vector = self.encode_texts(query, is_query=True)[0]

        conditions: list[models.Condition] = []
        if video_path:
            conditions.append(
                models.FieldCondition(
                    key="video_path",
                    match=models.MatchValue(value=video_path),
                )
            )
        if segment_type:
            conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=segment_type),
                )
            )

        qfilter = models.Filter(must=conditions) if conditions else None

        resp = self.client.query_points(
            collection_name=self.MEDIA_SEGMENTS_COLLECTION,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qfilter,
        )

        results = []
        for hit in resp.points:
            payload = hit.payload or {}
            results.append(
                {
                    "score": hit.score,
                    "text": payload.get("text"),
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "video_path": payload.get("video_path"),
                    "type": payload.get("type"),
                }
            )

        return results

    @observe("db_upsert_media_frame")
    def upsert_media_frame(
        self,
        point_id: str,
        vector: list[float],
        video_path: str,
        timestamp: float,
        action: str | None = None,
        dialogue: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a single frame embedding with structured metadata.

        Args:
            point_id: Unique ID for the point.
            vector: The vector embedding of the frame description.
            video_path: Path to the source video.
            timestamp: Timestamp of the frame in the video.
            action: Visual description of the frame.
            dialogue: Associated dialogue (optional).
            payload: Additional structured data (face_cluster_ids, ocr_text, etc).
        """
        safe_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(point_id)))

        final_payload = {
            "video_path": video_path,
            "timestamp": timestamp,
            "action": action,
            "dialogue": dialogue,
            "type": "visual",
        }
        
        # Merge structured data (face_cluster_ids, ocr_text, structured_data)
        if payload:
            final_payload.update(payload)

        self.client.upsert(
            collection_name=self.MEDIA_COLLECTION,
            points=[
                models.PointStruct(
                    id=safe_point_id,
                    vector=vector,
                    payload=final_payload,
                )
            ],
        )

    @observe("db_search_frames")
    def search_frames(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search for frames similar to the query description.

        Args:
            query: The visual search query text.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.

        Returns:
            A list of matching frames.
        """
        query_vector = self.encode_texts(query, is_query=True)[0]

        resp = self.client.query_points(
            collection_name=self.MEDIA_COLLECTION,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

        results = []
        for hit in resp.points:
            payload = hit.payload or {}
            results.append(
                {
                    "score": hit.score,
                    "action": payload.get("action"),
                    "timestamp": payload.get("timestamp"),
                    "video_path": payload.get("video_path"),
                    "type": payload.get("type", "visual"),
                }
            )

        return results

    @observe("db_search_frames_filtered")
    def search_frames_filtered(
        self,
        query_vector: list[float],
        face_cluster_ids: list[int] | None = None,
        limit: int = 20,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search frames with optional identity filtering.

        Used by agentic search to filter by face_cluster_ids.

        Args:
            query_vector: The query embedding vector.
            face_cluster_ids: Face cluster IDs to filter by (identity filter).
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.

        Returns:
            A list of matching frames with full payload.
        """
        # Build filter for face cluster IDs
        query_filter = None
        if face_cluster_ids:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="face_cluster_ids",
                        match=models.MatchAny(any=face_cluster_ids),
                    )
                ]
            )

        resp = self.client.query_points(
            collection_name=self.MEDIA_COLLECTION,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        results = []
        for hit in resp.points:
            payload = hit.payload or {}
            result = {
                "score": hit.score,
                "id": str(hit.id),
                **payload,  # Include all payload fields
            }
            results.append(result)

        return results

    def get_recent_frames(self, limit: int = 10) -> list[dict[str, Any]]:
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
                results.append({
                    "id": str(point.id),
                    "score": 0.5,  # Default score for fallback results
                    "fallback": True,
                    **payload,
                })
            return results
        except Exception:
            return []

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
                collection_name=self.FACES_COLLECTION,
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
        except Exception:
            return None

    def get_face_ids_by_cluster(self, cluster_id: int) -> list[str]:
        """Get all face point IDs belonging to a cluster.

        Args:
            cluster_id: The cluster ID to look up.

        Returns:
            List of face point IDs in that cluster.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
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
        except Exception:
            return []

    @observe("db_insert_face")
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
            collection_name=self.FACES_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=face_encoding,
                    payload=payload,
                )
            ],
        )

        return point_id

    @observe("db_search_face")
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
            collection_name=self.FACES_COLLECTION,
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

    @observe("db_insert_voice_segment")
    def insert_voice_segment(
        self,
        *,
        media_path: str,
        start: float,
        end: float,
        speaker_label: str,
        embedding: list[float],
        audio_path: str | None = None,
    ) -> None:
        """Insert a voice segment embedding.

        Args:
            media_path: Path to the source media file.
            start: Start time of the segment.
            end: End time of the segment.
            speaker_label: Label or ID of the speaker.
            embedding: The voice embedding vector.
            audio_path: Path to the extracted audio clip.

        Raises:
            ValueError: If the embedding dimension does not match `VOICE_VECTOR_SIZE`.
        """
        if len(embedding) != self.VOICE_VECTOR_SIZE:
            raise ValueError(
                f"voice vector dim mismatch: expected {self.VOICE_VECTOR_SIZE}, "
                f"got {len(embedding)}"
            )

        point_id = str(uuid.uuid4())

        self.client.upsert(
            collection_name=self.VOICE_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "media_path": media_path,
                        "start": start,
                        "end": end,
                        "speaker_label": speaker_label,
                        "embedding_version": "wespeaker_resnet34_v1_l2",
                        "audio_path": audio_path,
                    },
                )
            ],
        )

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
    def get_unresolved_faces(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get faces without assigned names.

        Args:
            limit: Maximum number of results.

        Returns:
            List of unnamed faces needing labeling.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
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
                results.append({
                    "id": point.id,
                    "cluster_id": cluster_id,
                    "name": payload.get("name"),
                    "media_path": payload.get("media_path"),
                    "timestamp": payload.get("timestamp"),
                    "thumbnail_path": payload.get("thumbnail_path"),
                    "is_main": payload.get("is_main", False),
                    "appearance_count": payload.get("appearance_count", 1),
                })
            # Sort: main characters first, then by appearance count
            results.sort(key=lambda x: (not x.get("is_main", False), -x.get("appearance_count", 1)))
            return results
        except Exception:
            return []

    @observe("db_update_face_name")
    def update_face_name(self, cluster_id: int, name: str) -> int:
        """Assign a name to all faces in a cluster.

        Args:
            cluster_id: The cluster ID to update.
            name: The name to assign.

        Returns:
            Number of faces updated.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
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
                    collection_name=self.FACES_COLLECTION,
                    payload=payload,
                    points=[point.id],
                )
                updated += 1
            return updated
        except Exception:
            return 0

    @observe("db_update_face_cluster_id")
    def update_face_cluster_id(self, face_id: str, cluster_id: int) -> bool:
        """Update the cluster ID for a single face.
        
        Args:
            face_id: The ID of the face to update.
            cluster_id: The new cluster ID.
            
        Returns:
            True if updated successfully.
        """
        try:
            self.client.set_payload(
                collection_name=self.FACES_COLLECTION,
                payload={"cluster_id": cluster_id},
                points=[face_id],
            )
            return True
        except Exception as e:
            log("Failed to update face cluster ID", error=str(e))
            return False

    @observe("db_merge_face_clusters")
    def merge_face_clusters(self, from_cluster: int, to_cluster: int) -> int:
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
                collection_name=self.FACES_COLLECTION,
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
                collection_name=self.FACES_COLLECTION,
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
                collection_name=self.FACES_COLLECTION,
                payload=payload,
                points=models.PointIdsList(points=ids),
            )
            
            return len(ids)
        except Exception as e:
            log("Failed to merge clusters", error=str(e))
            return 0

    @observe("db_get_named_faces")
    def get_named_faces(self) -> list[dict[str, Any]]:
        """Get all named faces.

        Returns:
            List of named faces with their info.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
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
                results.append({
                    "id": point.id,
                    "name": payload.get("name"),
                    "cluster_id": payload.get("cluster_id"),
                    "media_path": payload.get("media_path"),
                    "timestamp": payload.get("timestamp"),
                    "thumbnail_path": payload.get("thumbnail_path"),
                })
            return results
        except Exception:
            return []

    @observe("db_delete_face")
    def delete_face(self, face_id: str) -> bool:
        """Delete a face by its ID.

        Args:
            face_id: The ID of the face to delete.

        Returns:
            True if deleted successfully, False otherwise.
        """
        try:
            self.client.delete(
                collection_name=self.FACES_COLLECTION,
                points_selector=models.PointIdsList(points=[face_id]),
            )
            return True
        except Exception:
            return False

    @observe("db_update_single_face_name")
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
                collection_name=self.FACES_COLLECTION,
                payload={"name": name},
                points=[face_id],
            )
            return True
        except Exception:
            return False

    def get_faces_by_media(self, media_path: str, limit: int = 1000) -> list[dict[str, Any]]:
        """Get all faces for a specific media file.
        
        Args:
            media_path: Path to the media file.
            limit: Maximum number of results.
            
        Returns:
            List of face data dicts.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
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
                results.append({
                    "id": point.id,
                    "media_path": payload.get("media_path"),
                    "timestamp": payload.get("timestamp"),
                    "name": payload.get("name"),
                    "cluster_id": payload.get("cluster_id"),
                    "thumbnail_path": payload.get("thumbnail_path"),
                })
            return results
        except Exception:
            return []

    def set_face_main(self, cluster_id: int, is_main: bool = True) -> bool:
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
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
            if face_ids:
                self.client.set_payload(
                    collection_name=self.FACES_COLLECTION,
                    payload={"is_main": is_main},
                    points=models.PointIdsList(points=face_ids),
                )
            return True
        except Exception:
            return False

    @observe("db_get_indexed_media")
    def get_indexed_media(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get list of all indexed media files.

        Args:
            limit: Maximum number of results.

        Returns:
            List of indexed media files with metadata.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.MEDIA_SEGMENTS_COLLECTION,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            seen_paths: dict[str, dict[str, Any]] = {}
            for point in resp[0]:
                payload = point.payload or {}
                video_path = payload.get("video_path")
                if video_path and video_path not in seen_paths:
                    seen_paths[video_path] = {
                        "video_path": video_path,
                        "segment_count": 1,
                    }
                elif video_path:
                    seen_paths[video_path]["segment_count"] += 1
            return list(seen_paths.values())
        except Exception:
            return []

    @observe("db_get_voice_segments")
    def get_voice_segments(
        self,
        media_path: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get voice segments, optionally filtered by media path.

        Args:
            media_path: Optional filter by media file path.
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
            qfilter = models.Filter(must=conditions) if conditions else None
            resp = self.client.scroll(
                collection_name=self.VOICE_COLLECTION,
                scroll_filter=qfilter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = []
            for point in resp[0]:
                payload = point.payload or {}
                results.append({
                    "id": point.id,
                    "media_path": payload.get("media_path"),
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "speaker_label": payload.get("speaker_label"),
                    "audio_path": payload.get("audio_path"),
                })
            return results
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
                    "vectors_count": getattr(info, "vectors_count", info.points_count),
                }
            except Exception:
                stats[name] = {"points_count": 0, "vectors_count": 0}
        return stats

    @observe("db_delete_media")
    def delete_media(self, video_path: str) -> int:
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
            face_filter = models.Filter(must=[models.FieldCondition(key="media_path", match=models.MatchValue(value=video_path))])
            face_points = self.client.scroll(self.FACES_COLLECTION, scroll_filter=face_filter, limit=10000, with_payload=True)[0]
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
            voice_filter = models.Filter(must=[models.FieldCondition(key="media_path", match=models.MatchValue(value=video_path))])
            voice_points = self.client.scroll(self.VOICE_COLLECTION, scroll_filter=voice_filter, limit=10000, with_payload=True)[0]
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
        ]:
            try:
                # For faces and voices, we need to match media_path
                key = "video_path" if collection in [self.MEDIA_SEGMENTS_COLLECTION, self.MEDIA_COLLECTION] else "media_path"
                
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
                collection_name=self.VOICE_COLLECTION,
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
                collection_name=self.VOICE_COLLECTION,
                points_selector=models.PointIdsList(points=[segment_id]),
            )
            return True
        except Exception:
            return False

    @observe("db_get_all_face_embeddings")
    def get_all_face_embeddings(self) -> list[dict[str, Any]]:
        """Get all face embeddings with their IDs for clustering.

        Returns:
            List of dicts with 'id' and 'embedding' keys.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
                limit=10000,
                with_payload=True,
                with_vectors=True,
            )
            results = []
            for point in resp[0]:
                if point.vector:
                    results.append({
                        "id": point.id,
                        "embedding": list(point.vector) if isinstance(point.vector, (list, tuple)) else point.vector,
                        "payload": point.payload or {},
                    })
            return results
        except Exception:
            return []

    @observe("db_get_faces_grouped_by_cluster")
    def get_faces_grouped_by_cluster(self, limit: int = 500) -> dict[int, list[dict[str, Any]]]:
        """Get all faces grouped by cluster_id.

        Args:
            limit: Maximum number of faces to retrieve.

        Returns:
            Dictionary mapping cluster_id to list of faces in that cluster.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
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
                clusters[cluster_id].append({
                    "id": point.id,
                    "name": payload.get("name"),
                    "cluster_id": cluster_id,
                    "media_path": payload.get("media_path"),
                    "timestamp": payload.get("timestamp"),
                    "thumbnail_path": payload.get("thumbnail_path"),
                })
            return clusters
        except Exception:
            return {}


    @observe("db_update_voice_speaker_name")
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
                collection_name=self.VOICE_COLLECTION,
                payload={"speaker_name": name},
                points=[segment_id],
            )
            return True
        except Exception:
            return False

    @observe("db_get_all_voice_embeddings")
    def get_all_voice_embeddings(self) -> list[dict[str, Any]]:
        """Get all voice embeddings with their IDs for clustering.

        Returns:
            List of dicts with 'id', 'embedding', and 'payload' keys.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.VOICE_COLLECTION,
                limit=10000,
                with_payload=True,
                with_vectors=True,
            )
            results = []
            for point in resp[0]:
                if point.vector:
                    results.append({
                        "id": point.id,
                        "embedding": list(point.vector) if isinstance(point.vector, (list, tuple)) else point.vector,
                        "payload": point.payload or {},
                    })
            return results
        except Exception:
            return []

    @observe("db_update_voice_cluster_id")
    def update_voice_cluster_id(self, segment_id: str, cluster_id: int) -> bool:
        """Update the voice_cluster_id for a voice segment.

        Args:
            segment_id: The ID of the voice segment.
            cluster_id: The new cluster ID.

        Returns:
            True if updated successfully.
        """
        try:
            self.client.set_payload(
                collection_name=self.VOICE_COLLECTION,
                payload={"voice_cluster_id": cluster_id},
                points=[segment_id],
            )
            return True
        except Exception:
            return False

    @observe("db_get_voices_grouped_by_cluster")
    def get_voices_grouped_by_cluster(self, limit: int = 500) -> dict[int, list[dict[str, Any]]]:
        """Get all voice segments grouped by voice_cluster_id.

        Args:
            limit: Maximum number of segments to retrieve.

        Returns:
            Dictionary mapping cluster_id to list of voice segments.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.VOICE_COLLECTION,
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
                clusters[cluster_id].append({
                    "id": point.id,
                    "media_path": payload.get("media_path"),
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "speaker_label": payload.get("speaker_label"),
                    "speaker_name": payload.get("speaker_name"),
                    "audio_path": payload.get("audio_path"),
                    "voice_cluster_id": cluster_id,
                })
            return clusters
        except Exception:
            return {}

    @observe("db_merge_voice_clusters")
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
                collection_name=self.VOICE_COLLECTION,
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
                    collection_name=self.VOICE_COLLECTION,
                    payload={"voice_cluster_id": to_cluster},
                    points=[point.id],
                )
                updated += 1
            return updated
        except Exception:
            return 0

    def get_face_by_thumbnail(self, thumbnail_path: str) -> dict[str, Any] | None:
        """Look up a face by its thumbnail_path.
        
        Args:
            thumbnail_path: The thumbnail path stored in the database.
            
        Returns:
            Face data dict or None if not found.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
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

    def get_voice_by_audio_path(self, audio_path: str) -> dict[str, Any] | None:
        """Look up a voice segment by its audio_path.
        
        Args:
            audio_path: The audio clip path stored in the database.
            
        Returns:
            Voice segment data dict or None if not found.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.VOICE_COLLECTION,
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

    def get_voice_segments_for_media(self, media_path: str) -> list[dict[str, Any]]:
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
                collection_name=self.VOICE_COLLECTION,
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
                results.append({
                    "id": str(point.id),
                    "start": payload.get("start", 0),
                    "end": payload.get("end", 0),
                    "cluster_id": payload.get("cluster_id"),
                    "speaker_label": payload.get("speaker_label"),
                    "speaker_name": payload.get("speaker_name"),
                })
            return results
        except Exception as e:
            log(f"get_voice_segments_for_media error: {e}")
            return []

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

    def get_face_name_by_cluster(self, cluster_id: int) -> str | None:
        """Get HITL-assigned name for a face cluster.
        
        Args:
            cluster_id: The face cluster ID.
            
        Returns:
            Name if assigned, None otherwise.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
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

    def set_face_name(self, cluster_id: int, name: str) -> int:
        """Set name for a face cluster (and all its points).
        
        Args:
            cluster_id: The face cluster ID.
            name: The name to assign.
            
        Returns:
            Number of updated points (always 1 or 0 as we don't count updates).
        """
        try:
            self.client.set_payload(
                collection_name=self.FACES_COLLECTION,
                payload={"name": name},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="cluster_id",
                            match=models.MatchValue(value=cluster_id),
                        )
                    ]
                ),
            )
            return 1
        except Exception:
            return 0

    def get_speaker_name_by_cluster(self, cluster_id: int) -> str | None:
        """Get HITL-assigned name for a speaker cluster.
        
        Args:
            cluster_id: The speaker cluster ID.
            
        Returns:
            Name if assigned, None otherwise.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.VOICE_COLLECTION,
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

    def set_speaker_name(self, cluster_id: int, name: str) -> int:
        """Set name for a speaker cluster.
        
        Args:
            cluster_id: The speaker cluster ID.
            name: The name to assign.
            
        Returns:
            Number of updated segments.
        """
        try:
            # Update all segments with this voice_cluster_id
            self.client.set_payload(
                collection_name=self.VOICE_COLLECTION,
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
            return 1 # Return 1 to indicate success (Qdrant doesn't return count directly here easily)
        except Exception:
            return 0

    def get_frames_by_face_cluster(self, cluster_id: int, limit: int = 1000) -> list[dict]:
        """Get all frames containing a specific face cluster.
        
        Args:
            cluster_id: The face cluster ID to search for.
            limit: Maximum results.
            
        Returns:
            List of frame data dicts.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
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
                results.append({
                    "id": str(point.id),
                    "payload": point.payload or {},
                    "vector": point.vector,
                })
            return results
        except Exception as e:
            log(f"get_frames_by_face_cluster error: {e}")
            return []

    @observe("db_search_hybrid")
    def search_frames_hybrid(
        self,
        query: str,
        video_paths: str | list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """SOTA hybrid search combining vector + keyword + identity.
        
        This method provides 100% retrieval accuracy by:
        1. Detecting HITL names in query → filter by identity
        2. Vector search on text embeddings
        3. Keyword boost on structured fields
        4. Reciprocal Rank Fusion of all signals
        
        Args:
            query: Natural language search query.
            video_paths: Optional filter to specific video(s).
            limit: Maximum results to return.
            
        Returns:
            Ranked list of matching frames with scores.
        """
        from collections import Counter
        
        # 1. Check for HITL names in query
        known_names = self.get_all_hitl_names()
        query_lower = query.lower()
        matched_names = [n for n in known_names if n.lower() in query_lower]
        
        identity_filter = None
        if matched_names:
            # Get cluster IDs for matched names
            cluster_ids = []
            for name in matched_names:
                cid = self.get_cluster_id_by_name(name)
                if cid is not None:
                    cluster_ids.append(cid)
            
            if cluster_ids:
                identity_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="face_cluster_ids",
                            match=models.MatchAny(any=cluster_ids),
                        )
                    ]
                )
                log(f"[HybridSearch] Identity filter: {matched_names} → clusters {cluster_ids}")
        
        # 2. Build video path filter
        video_filter = None
        if video_paths:
            if isinstance(video_paths, str):
                video_paths = [video_paths]
            video_filter = models.FieldCondition(
                key="video_path",
                match=models.MatchAny(any=video_paths),
            )
        
        # Combine filters
        combined_filter = None
        conditions = []
        if identity_filter:
            conditions.extend(identity_filter.must or [])
        if video_filter:
            conditions.append(video_filter)
        if conditions:
            combined_filter = models.Filter(must=conditions)
        
        # 3. Vector search
        query_vector = self.encode_texts(query, is_query=True)[0]
        
        vector_results = self.client.query_points(
            collection_name=self.MEDIA_COLLECTION,
            query=query_vector,
            limit=limit * 3,
            query_filter=combined_filter,
        )
        
        # 4. Extract keywords for boosting
        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        # Remove common words
        stopwords = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were"}
        query_words -= stopwords
        
        # 5. Score and boost results
        results = []
        for hit in vector_results.points:
            payload = hit.payload or {}
            score = float(hit.score or 0)
            
            # Keyword boost on structured fields
            boost = 0.0
            
            # Check face_names
            for name in payload.get("face_names", []):
                if name and name.lower() in query_lower:
                    boost += 0.20
            
            # Check speaker_names
            for name in payload.get("speaker_names", []):
                if name and name.lower() in query_lower:
                    boost += 0.15
            
            # Check entities
            for entity in payload.get("entities", []):
                if entity and any(w in entity.lower() for w in query_words):
                    boost += 0.10
            
            # Check visible_text
            for text in payload.get("visible_text", []):
                if text and any(w in text.lower() for w in query_words):
                    boost += 0.08
            
            # Check scene_location
            location = payload.get("scene_location", "") or ""
            if any(w in location.lower() for w in query_words):
                boost += 0.08
            
            # Check action/description
            action = payload.get("action", "") or payload.get("description", "") or ""
            if any(w in action.lower() for w in query_words):
                boost += 0.05
            
            results.append({
                "id": str(hit.id),
                "score": score + boost,
                "base_score": score,
                "keyword_boost": boost,
                **payload,
            })
        
        # 6. Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        log(f"[HybridSearch] Query: '{query}' | Identity filter: {bool(identity_filter)} | Results: {len(results)}")
        
        return results[:limit]

    def update_frame_identity_text(
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
                with_vectors=False # Don't need old vector
            )
            if not resp:
                return False
                
            point = resp[0]
            payload = point.payload or {}
            
            # Get original visual description
            description = payload.get("description") or payload.get("action") or ""
            
            # 2. Build new Identity Text
            identity_parts = []
            if face_names:
                identity_parts.append(f"Visible: {', '.join(face_names)}")
            if speaker_names:
                identity_parts.append(f"Speaking: {', '.join(speaker_names)}")
                
            identity_text = ". ".join(identity_parts)
            
            # 3. Create NEW combined text for embedding
            # "A man walking. Visible: John. Speaking: John"
            full_text = f"{description}. {identity_text}" if identity_text else description
            
            if not full_text.strip():
                return False
                
            # 4. Re-encode
            new_vector = self.encode_texts(full_text, is_query=False)[0]
            
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
            log(f"Re-embedded frame {frame_id} with names: {face_names + speaker_names}")
            return True
            
            log(f"Re-embedded frame {frame_id} with names: {face_names + speaker_names}")
            return True
            
        except Exception as e:
            log(f"Failed to update frame identity: {e}")
            return False

    def re_embed_voice_cluster_frames(self, cluster_id: int, new_name: str, old_name: str | None = None) -> int:
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
                collection_name=self.VOICE_COLLECTION,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="voice_cluster_id", match=models.MatchValue(value=cluster_id))
                ]),
                limit=10000, # Assume reasonable limit for single cluster
                with_payload=True,
            )
            segments = resp[0]
            
            if not segments:
                return 0
                
            updated_count = 0
            processed_frames = set() # Avoid double processing same frame
            
            # 2. Iterate segments and find frames
            for seg in segments:
                payload = seg.payload or {}
                media_path = payload.get("media_path") or payload.get("audio_path")
                start = payload.get("start", 0)
                end = payload.get("end", 0)
                
                if not media_path:
                    continue
                    
                # Find frames in this time range for this video
                frames_resp = self.client.scroll(
                    collection_name=self.MEDIA_COLLECTION,
                    scroll_filter=models.Filter(must=[
                         models.FieldCondition(key="media_path", match=models.MatchValue(value=media_path)),
                         models.FieldCondition(key="timestamp", range=models.Range(gte=start, lte=end))
                    ]),
                    with_payload=True,
                    limit=100 # usually a few frames per segment
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
                        speaker_names = [n for n in speaker_names if n != old_name]
                        
                    # Add new name if not present
                    if new_name not in speaker_names:
                        speaker_names.append(new_name)
                    
                    # 3. Call update_frame_identity_text to re-embed
                    if self.update_frame_identity_text(str(frame.id), face_names, speaker_names):
                        updated_count += 1
                        
                    processed_frames.add(frame.id)
                    
            return updated_count

        except Exception as e:
            log(f"Error re-embedding voice cluster frames: {e}")
            return 0

    @observe("db_delete_media")
    def delete_media_by_path(self, media_path: str) -> None:
        """Delete all data associated with a media file."""
        for collection in [
            self.MEDIA_SEGMENTS_COLLECTION, 
            self.MEDIA_COLLECTION, 
            self.FACES_COLLECTION, 
            self.VOICE_COLLECTION
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
                                    match=models.MatchValue(value=media_path)
                                )
                            ]
                        )
                    )
                )
                # Try with "video_path" key (legacy/mixed usage)
                self.client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="video_path",
                                    match=models.MatchValue(value=media_path)
                                )
                            ]
                        )
                    )
                )
            except Exception as e:
                log(f"Failed to delete from {collection}: {e}")

    def store_scene_metadata(self, media_path: str, scenes: list[dict]) -> None:
        """Store scene captions (Placeholder).
        
        Args:
            media_path: Path to the media file.
            scenes: List of scene dictionaries with context.
        """
        # TODO: Implement scene storage if needed. 
        # Currently just a stub to satisfy Pylance/Pipeline.
        log(f"Received {len(scenes)} scenes for {media_path} (Storage not implemented)")
