"""Vector database interface for multimodal embeddings.

This module provides the `VectorDB` class, which handles interactions with Qdrant
for storing and retrieving media segments, frames, faces, and voice embeddings.
"""
from __future__ import annotations

import uuid
from pathlib import Path
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
    FACE_VECTOR_SIZE = 512  # InsightFace ArcFace (fallback SFace uses 128 but vectors are padded)
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
        media_path: str | None = None,
        timestamp: float | None = None,
        thumbnail_path: str | None = None,
    ) -> str:
        """Insert a face embedding.

        Args:
            face_encoding: The numeric vector representing the face.
            name: Name of the person (if known).
            cluster_id: ID of the cluster this face belongs to.
            media_path: Source media file path.
            timestamp: Timestamp in the video where face was detected.
            thumbnail_path: Path to the face thumbnail image.

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
                # Use point id hash as fallback cluster_id if missing
                if cluster_id is None:
                    cluster_id = abs(hash(str(point.id))) % (10**9)
                results.append({
                    "id": point.id,
                    "cluster_id": cluster_id,
                    "name": payload.get("name"),
                    "media_path": payload.get("media_path"),
                    "timestamp": payload.get("timestamp"),
                    "thumbnail_path": payload.get("thumbnail_path"),
                })
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
            if resp_target[0]:
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
                points=ids,
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

    @observe("db_update_face_cluster_id")
    def update_face_cluster_id(self, face_id: str, cluster_id: int) -> bool:
        """Update the cluster_id for a single face.

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
        except Exception:
            return False

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

    @observe("db_merge_face_clusters")
    def merge_face_clusters(self, from_cluster: int, to_cluster: int) -> int:
        """Merge two face clusters into one.

        Args:
            from_cluster: The cluster ID to merge from (will be removed).
            to_cluster: The cluster ID to merge into.

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
                    collection_name=self.FACES_COLLECTION,
                    payload={"cluster_id": to_cluster},
                    points=[point.id],
                )
                updated += 1
            return updated
        except Exception:
            return 0

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

