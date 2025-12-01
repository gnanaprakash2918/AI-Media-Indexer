import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


class VectorDB:
    MEDIA_SEGMENTS_COLLECTION = "media_segments"
    MEDIA_COLLECTION = "media_frames"
    FACES_COLLECTION = "faces"
    DOCS_COLLECTION = "docs"

    # all-MiniLM-L6-v2 output dim
    MEDIA_VECTOR_SIZE = 768
    FACE_VECTOR_SIZE = 128

    # encoder dim
    TEXT_DIM = 384

    def __init__(
        self,
        backend: str = "memory",
        host: str = "localhost",
        port: int = 6333,
        path: str = "qdrant_data",
    ):
        if backend == "memory":
            self.client = QdrantClient(path=path)
            print(f"Initialized embedded Qdrant at path={path}")
        elif backend == "docker":
            try:
                self.client = QdrantClient(host=host, port=port)
                self.client.get_collections()
            except Exception as e:
                print(f"Could not connect to Qdrant: {e}")
                raise ConnectionError("Qdrant connection failed.") from e
            print(f"Connected to Qdrant at {host}:{port}")
        else:
            raise ValueError(f"Unknown backend: {backend!r} (use 'memory' or 'docker')")

        print("Loading SentenceTransformer model...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        self._ensure_collections()

    def _ensure_collections(self) -> None:
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

    def list_collections(self):
        return self.client.get_collections()

    def insert_media_segments(
        self, video_path: str, segments: list[dict[str, Any]]
    ) -> None:
        if not segments:
            return

        print(f"Encoding {len(segments)} segments for {video_path}...")

        texts = [s.get("text", "") for s in segments]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        if len(embeddings[0]) != self.TEXT_DIM:
            raise ValueError(
                f"Text embedding dimension mismatch: expected {self.TEXT_DIM}, "
                f"got {len(embeddings[0])}"
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
                )
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
        query_vector = self.encoder.encode(query).tolist()

        conditions: list[models.Condition] = []

        if video_path is not None:
            conditions.append(
                models.FieldCondition(
                    key="video_path",
                    match=models.MatchValue(value=video_path),
                )
            )

        if segment_type is not None:
            conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=segment_type),
                )
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
                }
            )

        return results
