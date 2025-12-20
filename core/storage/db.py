"""Vector database interface for multimodal embeddings.

This module provides the `VectorDB` class, which handles interactions with Qdrant
for storing and retrieving media segments, frames, faces, and voice embeddings.
"""
from __future__ import annotations

import uuid
from typing import Any

import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from config import settings
from core.utils.logger import log
from core.utils.observe import observe


class VectorDB:
    """Wrapper for Qdrant vector database storage and retrieval."""

    MEDIA_SEGMENTS_COLLECTION = "media_segments"
    MEDIA_COLLECTION = "media_frames"
    FACES_COLLECTION = "faces"
    VOICE_COLLECTION = "voice_segments"

    MEDIA_VECTOR_SIZE = 384
    FACE_VECTOR_SIZE = 128
    TEXT_DIM = 384
    VOICE_VECTOR_SIZE = 256

    MODEL_NAME = "all-MiniLM-L6-v2"

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
            return SentenceTransformer(path_or_name, device=device)

        if local_model_dir.exists():
            log(
                "Loading SentenceTransformer from local cache",
                path=str(local_model_dir),
                device=target_device,
            )
            try:
                return _create(str(local_model_dir), device=target_device)
            except Exception as exc:
                log(
                    "Failed to load cached model on target device — fallback",
                    error=str(exc),
                )

        log(
            "Local model not found, loading from hub",
            model_name=self.MODEL_NAME,
            device=target_device,
        )

        try:
            model = _create(self.MODEL_NAME, device=target_device)
        except Exception as exc:
            log("Falling back to CPU encoder", error=str(exc))
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
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: A string or list of strings to encode.
            batch_size: Batch size for inference.
            show_progress_bar: Whether to show a tqdm progress bar.

        Returns:
            A list of vector embeddings (lists of floats).
        """
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

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
        try:
            if not self.client.collection_exists(self.MEDIA_SEGMENTS_COLLECTION):
                self.client.create_collection(
                    collection_name=self.MEDIA_SEGMENTS_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=self.TEXT_DIM,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as e:
            # Handle potential race condition or file exists error
            if "File exists" not in str(e) and "500" not in str(e):
                log("Warning: media_segments collection creation error", error=str(e))

        try:
            if not self.client.collection_exists(self.MEDIA_COLLECTION):
                self.client.create_collection(
                    collection_name=self.MEDIA_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=self.MEDIA_VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as e:
            if "File exists" not in str(e) and "500" not in str(e):
                log("Warning: media_frames collection creation error", error=str(e))

        try:
            if not self.client.collection_exists(self.FACES_COLLECTION):
                self.client.create_collection(
                    collection_name=self.FACES_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=self.FACE_VECTOR_SIZE,
                        distance=models.Distance.EUCLID,
                    ),
                )
        except Exception as e:
            if "File exists" not in str(e) and "500" not in str(e):
                log("Warning: faces collection creation error", error=str(e))

        try:
            if not self.client.collection_exists(self.VOICE_COLLECTION):
                self.client.create_collection(
                    collection_name=self.VOICE_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=self.VOICE_VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as e:
            if "File exists" not in str(e) and "500" not in str(e):
                log("Warning: voice_segments collection creation error", error=str(e))

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
        query_vector = self.encode_texts(query)[0]

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
    ) -> None:
        """Upsert a single frame embedding.

        Args:
            point_id: Unique ID for the point.
            vector: The vector embedding of the frame description.
            video_path: Path to the source video.
            timestamp: Timestamp of the frame in the video.
            action: Visual description of the frame.
            dialogue: Associated dialogue (optional).
        """
        safe_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(point_id)))

        payload = {
            "video_path": video_path,
            "timestamp": timestamp,
            "action": action,
            "dialogue": dialogue,
            "type": "visual",
        }

        self.client.upsert(
            collection_name=self.MEDIA_COLLECTION,
            points=[
                models.PointStruct(
                    id=safe_point_id,
                    vector=vector,
                    payload=payload,
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
        query_vector = self.encoder.encode(query).tolist()

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

    @observe("db_insert_face")
    def insert_face(
        self,
        face_encoding: list[float],
        name: str | None = None,
        cluster_id: int | None = None,
    ) -> str:
        """Insert a face embedding.

        Args:
            face_encoding: The numeric vector representing the face.
            name: Name of the person (if known).
            cluster_id: ID of the cluster this face belongs to.

        Returns:
            The generated ID of the inserted point.
        """
        point_id = str(uuid.uuid4())

        payload = {
            "name": name,
            "cluster_id": cluster_id,
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
    ) -> None:
        """Insert a voice segment embedding.

        Args:
            media_path: Path to the source media file.
            start: Start time of the segment.
            end: End time of the segment.
            speaker_label: Label or ID of the speaker.
            embedding: The voice embedding vector.

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
