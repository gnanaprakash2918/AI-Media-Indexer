"""Vector database utilities for media, faces using Qdrant."""

from __future__ import annotations

import uuid
from typing import Any

import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from config import settings
from core.utils.logger import log


class VectorDB:
    """Wrapper around Qdrant for storing and querying embeddings.

    Manages:
    - Embedded or remote Qdrant initialization
    - Collection setup
    - Media frame upsert
    - Face insertion and search
    - Local SentenceTransformer model loading

    Attributes:
        MEDIA_SEGMENTS_COLLECTION: Collection name for media text segments.
        MEDIA_COLLECTION: Collection name for media frame vectors.
        FACES_COLLECTION: Collection name for face embeddings.
        MEDIA_VECTOR_SIZE: Expected dimension of media frame vectors.
        FACE_VECTOR_SIZE: Expected dimension of face embeddings.
        TEXT_DIM: Expected dimension of text embeddings.
        MODEL_NAME: Sentence-transformers model identifier.
    """

    MEDIA_SEGMENTS_COLLECTION = "media_segments"
    MEDIA_COLLECTION = "media_frames"
    FACES_COLLECTION = "faces"

    MEDIA_VECTOR_SIZE = 384
    FACE_VECTOR_SIZE = 128
    TEXT_DIM = 384
    MODEL_NAME = "all-MiniLM-L6-v2"

    client: QdrantClient

    def __init__(
        self,
        backend: str = settings.qdrant_backend,
        host: str = settings.qdrant_host,
        port: int = settings.qdrant_port,
        path: str = "qdrant_data_embedded",
    ) -> None:
        """Initialize the vector database.

        Args:
            backend: Qdrant backend to use. Either "memory" (embedded) or "docker".
            host: Qdrant host for the "docker" backend.
            port: Qdrant port for the "docker" backend.
            path: Storage path for embedded Qdrant when using "memory".

        Raises:
            ConnectionError: If a Qdrant connection cannot be established.
            ValueError: If an unknown backend value is provided.
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
        """Load SentenceTransformer with fallback: try settings.device, else cpu.

        - tries local cache first, then hub.
        - attempts target device (settings.device), and falls back to CPU on failure.
        """
        models_dir = settings.model_cache_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        local_model_dir = models_dir / self.MODEL_NAME
        target_device = settings.device or "cpu"

        def _create(path_or_name: str, device: str) -> SentenceTransformer:
            log(
                "Creating SentenceTransformer", path_or_name=path_or_name, device=device
            )
            return SentenceTransformer(path_or_name, device=device)

        # Try local cache first
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
                    "Failed to load cached model on target device — will try hub or "
                    "CPU fallback",
                    error=str(exc),
                    device=target_device,
                )

        # Try loading from hub on target_device
        log(
            "Local model not found, loading from hub",
            path=str(local_model_dir),
            model_name=self.MODEL_NAME,
            device=target_device,
        )
        try:
            model = _create(self.MODEL_NAME, device=target_device)
        except Exception as exc:
            if target_device != "cpu":
                log(
                    "Falling back to CPU for SentenceTransformer after failure",
                    from_device=target_device,
                    error=str(exc),
                )
                try:
                    model = _create(self.MODEL_NAME, device="cpu")
                except Exception as exc2:
                    log(
                        "Failed to load SentenceTransformer on CPU as well",
                        error=str(exc2),
                    )
                    raise
            else:
                log("Failed to load SentenceTransformer on CPU", error=str(exc))
                raise

        try:
            log("Saving SentenceTransformer model to cache", path=str(local_model_dir))
            model.save(str(local_model_dir))
        except Exception as exc:  # noqa: BLE001
            log(
                "Warning: failed to save SentenceTransformer model",
                path=str(local_model_dir),
                error=str(exc),
            )

        return model

    def encode_texts(
        self,
        texts: str | list[str],
        batch_size: int = 1,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        """Encode a single text or list of texts in a memory-safe manner.

        - Empties CUDA cache before encoding attempts.
        - Uses batch_size=1 by default to reduce GPU peaks.
        - On encoding failure (e.g., CUDA OOM), automatically retries on CPU.
        - Returns embeddings as plain Python lists: List[List[float]].
        """
        # Normalize input to list
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        def _try_encode(encoder: SentenceTransformer, inputs: list[str], bsize: int):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                log("Warning: torch.cuda.empty_cache() failed", error=str(exc))

            embeddings = encoder.encode(
                inputs, batch_size=bsize, show_progress_bar=show_progress_bar
            )
            return [list(e) for e in embeddings]

        try:
            return _try_encode(self.encoder, texts_list, batch_size)
        except RuntimeError as exc:
            log("Encoder runtime error; attempting CPU fallback", error=str(exc))
        except Exception as exc:
            log("Unexpected encoder error; attempting CPU fallback", error=str(exc))

        try:
            need_new_cpu_encoder = False
            try:
                if settings.device and "cuda" in str(settings.device).lower():
                    need_new_cpu_encoder = True
            except Exception:
                need_new_cpu_encoder = True

            if need_new_cpu_encoder:
                log("Creating CPU fallback SentenceTransformer", device="cpu")
                cpu_encoder = SentenceTransformer(self.MODEL_NAME, device="cpu")
            else:
                cpu_encoder = self.encoder

            return _try_encode(cpu_encoder, texts_list, 1)
        except Exception as exc:
            log("CPU fallback encoding failed", error=str(exc))
            raise

    def _ensure_collections(self) -> None:
        """Ensure that required Qdrant collections exist."""
        if not self.client.collection_exists(self.MEDIA_SEGMENTS_COLLECTION):
            log("media_segments collection not found, creating")
            self.client.create_collection(
                collection_name=self.MEDIA_SEGMENTS_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.TEXT_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

        if not self.client.collection_exists(self.MEDIA_COLLECTION):
            log("media_frames collection not found, creating")
            self.client.create_collection(
                collection_name=self.MEDIA_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.MEDIA_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

        if not self.client.collection_exists(self.FACES_COLLECTION):
            log("faces collection not found, creating")
            self.client.create_collection(
                collection_name=self.FACES_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.FACE_VECTOR_SIZE,
                    distance=models.Distance.EUCLID,
                ),
            )

        log("Qdrant collections ensured")

    def list_collections(self) -> models.CollectionsResponse:
        """List all collections in Qdrant.

        Returns:
            Qdrant collections response.
        """
        return self.client.get_collections()

    def insert_media_segments(
        self,
        video_path: str,
        segments: list[dict[str, Any]],
    ) -> None:
        """Insert text segments for a media file into Qdrant.

        Args:
            video_path: Path to the media file.
            segments: List of segment dictionaries containing:
                - "text": Segment text.
                - "start": Start time in seconds.
                - "end": End time in seconds.
                - "type": Segment type (for example, "dialogue").
        """
        if not segments:
            return

        log(
            "Encoding media segments",
            video_path=video_path,
            segment_count=len(segments),
        )

        texts = [s.get("text", "") for s in segments]
        embeddings = self.encode_texts(texts, batch_size=1, show_progress_bar=False)

        if not embeddings or len(embeddings[0]) != self.TEXT_DIM:
            raise ValueError(
                f"Text embedding dimension mismatch: expected {self.TEXT_DIM}, "
                f"got {len(embeddings[0]) if embeddings else 'no embeddings'}",
            )

        points: list[models.PointStruct] = []
        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0.0)
            unique_str = f"{video_path}_{start_time}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
            payload = {
                "video_path": str(video_path),
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
                ),
            )

        self.client.upsert(
            collection_name=self.MEDIA_SEGMENTS_COLLECTION,
            points=points,
        )

        log(
            "Inserted media text segments into Qdrant",
            video_path=video_path,
            inserted_count=len(points),
        )

    def search_media(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
        video_path: str | None = None,
        segment_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for media text segments matching a query.

        Args:
            query: Search query text.
            limit: Maximum number of results to return.
            score_threshold: Optional similarity threshold.
            video_path: Optional exact match filter on the media file path.
            segment_type: Optional exact match filter on segment type.

        Returns:
            List of result dictionaries containing:
                - "score": Similarity score.
                - "text": Segment text.
                - "start": Start time in seconds.
                - "end": End time in seconds.
                - "video_path": Media file path.
                - "type": Segment type.
        """
        query_vector = self.encode_texts(query, batch_size=1, show_progress_bar=False)[
            0
        ]

        conditions: list[models.Condition] = []
        if video_path is not None:
            conditions.append(
                models.FieldCondition(
                    key="video_path",
                    match=models.MatchValue(value=video_path),
                ),
            )

        if segment_type is not None:
            conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=segment_type),
                ),
            )

        qdrant_filter = models.Filter(must=conditions) if conditions else None

        resp = self.client.query_points(
            collection_name=self.MEDIA_SEGMENTS_COLLECTION,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        hits = resp.points
        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "score": hit.score,
                    "text": payload.get("text"),
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "video_path": payload.get("video_path"),
                    "type": payload.get("type"),
                },
            )

        log(
            "Media text search completed",
            query=query,
            limit=limit,
            result_count=len(results),
        )
        return results

    def upsert_media_frame(
        self,
        point_id: str,
        vector: list[float],
        video_path: str,
        timestamp: float,
        action: str | None = None,
        dialogue: str | None = None,
    ) -> None:
        """Upsert a media frame embedding into Qdrant.

        Args:
            point_id: Identifier for the frame.
            vector: Embedding vector.
            video_path: Path to the media file.
            timestamp: Frame timestamp in seconds.
            action: Optional action description.
            dialogue: Optional dialogue text.

        Raises:
            ValueError: If the vector dimension is incorrect.
        """
        if len(vector) != self.MEDIA_VECTOR_SIZE:
            raise ValueError(
                f"media frame vector dim mismatch: expected {self.MEDIA_VECTOR_SIZE}, "
                f"got {len(vector)}",
            )

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
                ),
            ],
        )

        log(
            "[VectorDB] Upserted media frame",
            point_id=safe_point_id,
            video_path=video_path,
            timestamp=timestamp,
            has_action=action is not None,
            has_dialogue=dialogue is not None,
        )

    def search_frames(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search visual media frames using a text query.

        Args:
            query: Text query describing the desired visual content.
            limit: Maximum number of results to return.
            score_threshold: Optional similarity threshold.

        Returns:
            List of dictionaries containing:
                - score: Similarity score.
                - action: Detected action, if any.
                - timestamp: Frame timestamp in seconds.
                - video_path: Media file path.
                - type: Frame type (defaults to "visual").
        """
        query_vector = self.encoder.encode(query).tolist()

        resp = self.client.query_points(
            collection_name=self.MEDIA_COLLECTION,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

        hits = resp.points
        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "score": hit.score,
                    "action": payload.get("action"),
                    "timestamp": payload.get("timestamp"),
                    "video_path": payload.get("video_path"),
                    "type": payload.get("type", "visual"),
                },
            )

        log(
            "Media frame search completed",
            query=query,
            limit=limit,
            result_count=len(results),
        )
        return results

    def insert_face(
        self,
        face_encoding: list[float],
        name: str | None = None,
        cluster_id: int | None = None,
    ) -> str:
        """Insert a face embedding into the faces collection.

        Args:
            face_encoding: Face embedding vector.
            name: Optional human-readable label.
            cluster_id: Optional cluster identifier.

        Returns:
            Generated point ID.

        Raises:
            ValueError: If the vector dimension is incorrect.
        """
        if len(face_encoding) != self.FACE_VECTOR_SIZE:
            raise ValueError(
                f"face vector dim mismatch: expected {self.FACE_VECTOR_SIZE}, "
                f"got {len(face_encoding)}",
            )

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
                ),
            ],
        )

        log(
            "[VectorDB] Upserted face",
            point_id=point_id,
            name=name,
            cluster_id=cluster_id,
        )
        return point_id

    def search_face(
        self,
        face_encoding: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar faces in the faces collection.

        Args:
            face_encoding: Query face embedding vector.
            limit: Maximum number of results to return.
            score_threshold: Optional distance threshold.

        Returns:
            List of result dictionaries containing:
                - "score": Distance score.
                - "id": Point ID.
                - "name": Stored name, if any.
                - "cluster_id": Stored cluster ID, if any.

        Raises:
            ValueError: If the query vector dimension is incorrect.
        """
        if len(face_encoding) != self.FACE_VECTOR_SIZE:
            raise ValueError(
                f"face query vector dim mismatch: expected {self.FACE_VECTOR_SIZE}, "
                f"got {len(face_encoding)}",
            )

        resp = self.client.query_points(
            collection_name=self.FACES_COLLECTION,
            query=face_encoding,
            limit=limit,
            score_threshold=score_threshold,
        )

        hits = resp.points
        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "score": hit.score,
                    "id": hit.id,
                    "name": payload.get("name"),
                    "cluster_id": payload.get("cluster_id"),
                },
            )

        log(
            "Face search completed",
            limit=limit,
            result_count=len(results),
        )
        return results

    def close(self) -> None:
        """Close the underlying Qdrant client.

        Suppresses errors while closing; intended to be called explicitly at
        application shutdown.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self.client.close()
        except Exception as exc:  # noqa: BLE001
            log("Warning: error while closing QdrantClient", error=str(exc))


if __name__ == "__main__":
    db = VectorDB()
    try:
        db.insert_media_segments(
            "test_video.mp4",
            [
                {
                    "text": "Hello world this is a test",
                    "start": 0.0,
                    "end": 2.0,
                    "type": "dialogue",
                },
            ],
        )
        log(
            "Media search result",
            results=db.search_media(query="Hello world", limit=2),
        )
    finally:
        db.close()
