"""Vector database utilities for media, faces, and documents using Qdrant."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from config import Settings

_SETTINGS = Settings()


class VectorDB:
    """Wrapper around Qdrant for storing and querying embeddings.

    Manages:
    - Embedded or remote Qdrant initialization
    - Collection setup
    - Text segment insertion and search
    - Media frame upsert
    - Face insertion and search
    - Local SentenceTransformer model loading

    Attributes:
        MEDIA_SEGMENTS_COLLECTION: Collection name for media text segments.
        MEDIA_COLLECTION: Collection name for media frame vectors.
        FACES_COLLECTION: Collection name for face embeddings.
        DOCS_COLLECTION: Collection name for document embeddings.
        MEDIA_VECTOR_SIZE: Expected dimension of media frame vectors.
        FACE_VECTOR_SIZE: Expected dimension of face embeddings.
        TEXT_DIM: Expected dimension of text embeddings.
        MODEL_NAME: Sentence-transformers model identifier.
    """

    MEDIA_SEGMENTS_COLLECTION = "media_segments"
    MEDIA_COLLECTION = "media_frames"
    FACES_COLLECTION = "faces"
    DOCS_COLLECTION = "docs"

    MEDIA_VECTOR_SIZE = 768
    FACE_VECTOR_SIZE = 128
    TEXT_DIM = 384

    MODEL_NAME = "all-MiniLM-L6-v2"

    client: QdrantClient

    def __init__(
        self,
        backend: str = "memory",
        host: str = "localhost",
        port: int = 6333,
        path: str = "qdrant_data",
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
            print(f"Initialized embedded Qdrant at path={path}")
        elif backend == "docker":
            try:
                self.client = QdrantClient(host=host, port=port)
                self.client.get_collections()
            except Exception as exc:  # noqa: BLE001
                print(f"Could not connect to Qdrant: {exc}")
                raise ConnectionError("Qdrant connection failed.") from exc
            print(f"Connected to Qdrant at {host}:{port}")
        else:
            raise ValueError(f"Unknown backend: {backend!r} (use 'memory' or 'docker')")

        print("Loading SentenceTransformer model...")
        self.encoder = self._load_encoder()

        self._ensure_collections()

    @staticmethod
    def _get_project_root() -> Path:
        """Return the project root directory from application settings.

        Returns:
            Project root path discovered via configuration.
        """
        return _SETTINGS.project_root

    def _load_encoder(self) -> SentenceTransformer:
        """Load the SentenceTransformer model with project-local preference.

        Load order:
        1. `<project_root>/models/<MODEL_NAME>` if it exists.
        2. Global cache or download using `MODEL_NAME`.
        3. Save the loaded model into `<project_root>/models/<MODEL_NAME>`.

        Returns:
            Loaded SentenceTransformer encoder.
        """
        project_root = self._get_project_root()
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        local_model_dir = models_dir / self.MODEL_NAME

        if local_model_dir.exists():
            print(f"Loading SentenceTransformer from {local_model_dir}")
            return SentenceTransformer(str(local_model_dir), device="cpu")

        print(
            f"Local model not found in {local_model_dir}. "
            f"Loading '{self.MODEL_NAME}' from SentenceTransformers..."
        )
        model = SentenceTransformer(self.MODEL_NAME, device="cpu")

        try:
            print(f"Saving model to {local_model_dir}")
            model.save(str(local_model_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to save model to {local_model_dir}: {exc}")

        return model

    def _ensure_collections(self) -> None:
        """Ensure that required Qdrant collections exist."""
        if not self.client.collection_exists(self.MEDIA_SEGMENTS_COLLECTION):
            print("media_segments collection not found. Creating it.")
            self.client.create_collection(
                collection_name=self.MEDIA_SEGMENTS_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.TEXT_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

        if not self.client.collection_exists(self.MEDIA_COLLECTION):
            print("media_frames collection not found. Creating it.")
            self.client.create_collection(
                collection_name=self.MEDIA_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.MEDIA_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

        if not self.client.collection_exists(self.FACES_COLLECTION):
            print("faces collection not found. Creating it.")
            self.client.create_collection(
                collection_name=self.FACES_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.FACE_VECTOR_SIZE,
                    distance=models.Distance.EUCLID,
                ),
            )

        if not self.client.collection_exists(self.DOCS_COLLECTION):
            print("docs collection not found. Creating it.")
            self.client.create_collection(
                collection_name=self.DOCS_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.TEXT_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

        print("Collections ensured!")

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

        print(f"Encoding {len(segments)} segments for {video_path}...")

        texts = [s.get("text", "") for s in segments]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        if len(embeddings[0]) != self.TEXT_DIM:
            raise ValueError(
                f"Text embedding dimension mismatch: expected {self.TEXT_DIM}, "
                f"got {len(embeddings[0])}",
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
                    vector=embeddings[i].tolist(),
                    payload=payload,
                ),
            )

        self.client.upsert(
            collection_name=self.MEDIA_SEGMENTS_COLLECTION,
            points=points,
        )

        print(f"Inserted {len(points)} segments into Qdrant.")

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
        query_vector = self.encoder.encode(query).tolist()

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

        payload = {
            "video_path": video_path,
            "timestamp": timestamp,
            "action": action,
            "dialogue": dialogue,
        }

        self.client.upsert(
            collection_name=self.MEDIA_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                ),
            ],
        )

        print(f"[VectorDB] Upserted media frame id={point_id}")

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

        print(f"[VectorDB] Upserted face id={point_id}, name={name}")
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
            print(f"Warning: error while closing QdrantClient: {exc}")


if __name__ == "__main__":
    db = VectorDB(backend="memory")

    try:
        db.insert_media_segments(
            "C:\\Users\\Gnana Prakash M\\Downloads\\project\\test_video.mp4",
            [
                {
                    "text": "Hello world this is a test",
                    "start": 0.0,
                    "end": 2.0,
                    "type": "dialogue",
                },
            ],
        )

        print("Media search result (no filter):", db.search_media("Hello world"))
        print(
            "Media search result (filtered):",
            db.search_media(
                "Hello world",
                video_path="test_video.mp4",
                segment_type="dialogue",
            ),
        )

        fake_face_vec = [0.01] * VectorDB.FACE_VECTOR_SIZE
        db.insert_face(fake_face_vec, name="Test Person", cluster_id=0)

        face_results = db.search_face(fake_face_vec, limit=3)
        print("Face search result:", face_results)
    finally:
        db.close()
