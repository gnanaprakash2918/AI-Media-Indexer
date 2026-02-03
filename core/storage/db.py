"""Vector database interface for multimodal embeddings.

This module provides the `VectorDB` class, which handles interactions with Qdrant
for storing and retrieving media segments, frames, faces, and voice embeddings.
"""

from __future__ import annotations

import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from config import settings
from core.utils.hardware import select_embedding_model
from core.utils.logger import log
from core.utils.observe import observe
from core.storage.filters import (
    video_path_filter,
    build_filter,
)


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
                        log(
                            f"Qdrant connection error (attempt {attempt + 1}): {e}, retrying..."
                        )
                        time.sleep(delay * (attempt + 1))
                    else:
                        log(
                            f"Qdrant connection failed after {max_retries} attempts: {e}"
                        )
            if last_error:
                raise last_error
            raise RuntimeError("Qdrant retry failed")

        return wrapper

    return decorator


def sanitize_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types.

    Prevents 'Unable to serialize numpy.int32' errors when upserting to Qdrant.
    Also detects and handles unawaited coroutines which would cause serialization failures.
    """
    import asyncio
    import inspect

    import numpy as np

    # CRITICAL: Detect unawaited coroutines which would cause Pydantic serialization errors
    if asyncio.iscoroutine(obj) or inspect.iscoroutine(obj):
        log(
            f"[SANITIZE] ERROR: Unawaited coroutine detected in payload: {obj}. "
            "This indicates a missing 'await' somewhere in the pipeline. "
            "Replacing with error placeholder to prevent crash.",
            level="ERROR",
        )
        return "[ERROR: Unawaited coroutine - check logs]"

    if isinstance(obj, dict):
        return {k: sanitize_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# Auto-select embedding model based on available VRAM
# Allow override from config
if settings.embedding_model_override:
    _SELECTED_MODEL = settings.embedding_model_override
    log(f"Overriding embedding model from config: {_SELECTED_MODEL}")
else:
    _SELECTED_MODEL, _ = select_embedding_model()


def paginated_scroll(
    client,
    collection_name: str,
    scroll_filter: models.Filter | None = None,
    limit: int = 1000,
    batch_size: int = 100,
    with_payload: bool = True,
    with_vectors: bool = False,
) -> list:
    """Generator-based paginated scroll for large result sets.

    Prevents memory issues by fetching results in batches using Qdrant's
    cursor-based pagination instead of loading all at once.

    Args:
        client: Qdrant client instance.
        collection_name: Name of the collection to scroll.
        scroll_filter: Optional filter to apply.
        limit: Maximum total results to return.
        batch_size: Number of records per batch (default 100).
        with_payload: Whether to include payload in results.
        with_vectors: Whether to include vectors in results.

    Returns:
        List of all matching points up to limit.
    """
    all_points = []
    offset = None
    remaining = limit

    while remaining > 0:
        fetch_size = min(batch_size, remaining)

        result, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=fetch_size,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        all_points.extend(result)
        remaining -= len(result)

        if next_offset is None or len(result) < fetch_size:
            break

        offset = next_offset

    return all_points


class VectorDB:
    """Wrapper for Qdrant vector database storage and retrieval."""

    MEDIA_SEGMENTS_COLLECTION = "media_segments"
    MEDIA_COLLECTION = "media_frames"
    FRAMES_COLLECTION = "media_frames"  # Alias for summarizer
    FACES_COLLECTION = "faces"
    VOICE_COLLECTION = "voice_segments"
    SCENES_COLLECTION = "scenes"  # Scene-level storage (production approach)
    SCENELETS_COLLECTION = (
        "media_scenelets"  # Sliding window (5s) for action search
    )
    SUMMARIES_COLLECTION = "global_summaries"  # Hierarchical summaries (L1/L2)
    MASKLETS_COLLECTION = "masklets"
    AUDIO_EVENTS_COLLECTION = "audio_events"
    VIDEO_METADATA_COLLECTION = "video_metadata"

    MEDIA_VECTOR_SIZE = settings.visual_embedding_dim
    FACE_VECTOR_SIZE = 512  # InsightFace ArcFace

    # DYNAMICALLY FIX DIM: If model is bge-m3, we force 1024, otherwise use config
    if (
        "bge-m3" in _SELECTED_MODEL.lower()
        or "mxbai" in _SELECTED_MODEL.lower()
    ):
        TEXT_DIM = 1024
    else:
        TEXT_DIM = settings.text_embedding_dim

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

        # LAZY LOADING: Do NOT load encoder at startup to prevent OOM
        # Encoder will be loaded on first encode_texts() call
        self.encoder: SentenceTransformer | None = None
        self._encoder_last_used: float = 0.0
        self._idle_unload_seconds = 300  # Unload after 5 min idle

        log(
            f"VectorDB initialized (lazy mode). Encoder: {self.MODEL_NAME} will load on first use."
        )
        self._ensure_collections()  # This is sync and should stay sync as it's in __init__

        # Thread-safe cluster ID counter (uses timestamp + counter for uniqueness)
        import threading

        self._cluster_id_lock = threading.Lock()
        self._cluster_id_counter = 0

        # Expected dimensions for validation (Issue 9)
        self._expected_dims = {
            self.MEDIA_COLLECTION: self.MEDIA_VECTOR_SIZE,
            self.FACES_COLLECTION: self.FACE_VECTOR_SIZE,
            self.SCENE_COLLECTION: self.TEXT_DIM,
            self.SCENELETS_COLLECTION: self.TEXT_DIM,
            self.VOICE_COLLECTION: self.VOICE_VECTOR_SIZE,
        }

    def _validate_vector_dim(self, vector: list | None, collection: str, context: str = "") -> bool:
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
                level="ERROR"
            )
            return False
        return True

        # Embedding cache for repeated queries
        self._embedding_cache = {}
        self._embedding_cache_max_size = 1000

        # Load Visual Encoder for cross-modal search (Text -> Visual Embedding)
        from core.processing.visual_encoder import get_default_visual_encoder

        self.visual_encoder = get_default_visual_encoder()

    def get_next_voice_cluster_id(self) -> int:
        """Generate a unique voice cluster ID.

        Uses timestamp-based unique ID with atomic counter to guarantee uniqueness.
        Format: YYMMDDHHMM + 4-digit counter = 14-digit ID that's time-sortable.

        Returns:
            Unique integer cluster ID.
        """
        import time

        with self._cluster_id_lock:
            # Get timestamp component (minutes since epoch, fits in reasonable int)
            ts = int(time.time() // 60)  # Minutes since epoch
            self._cluster_id_counter = (self._cluster_id_counter + 1) % 10000
            # Combine: timestamp * 10000 + counter
            cluster_id = (ts % 100000000) * 10000 + self._cluster_id_counter
            return cluster_id

    def get_next_face_cluster_id(self) -> int:
        """Generate a unique face cluster ID.

        Uses same logic as voice cluster IDs for consistency.

        Returns:
            Unique integer cluster ID.
        """
        return self.get_next_voice_cluster_id()

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
                    collection_name=self.VOICE_COLLECTION,
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

    def get_max_face_cluster_id(self) -> int:
        """Get the maximum existing face cluster ID."""
        try:
            max_id = 0
            offset = None
            while True:
                results, offset = self.client.scroll(
                    collection_name=self.FACES_COLLECTION,
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

    def _load_encoder(self) -> SentenceTransformer:
        """Loads the SentenceTransformer model from local cache or HuggingFace Hub.

        Attempts to load from a local directory first. If missing or corrupt,
        it downloads the model using snapshot_download to ensure all necessary
        files are present. It also handles GPU-to-CPU fallback logic.

        Returns:
            A SentenceTransformer instance loaded on the configured device.
        """
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
            log(
                "Loading cached model",
                path=str(local_model_dir),
                device=target_device,
            )
            try:
                return _create(str(local_model_dir), device=target_device)
            except Exception as exc:
                log(
                    f"GPU Load Failed: {exc}. Retrying on CPU...",
                    level="warning",
                )
                try:
                    return _create(str(local_model_dir), device="cpu")
                except Exception as exc_cpu:
                    log(f"CPU Fallback Failed: {exc_cpu}", level="error")
                    # Local likely corrupt, fall through to re-download
                    pass

        log(
            "Local model missing/corrupt, downloading from Hub",
            model=self.MODEL_NAME,
        )

        # Use snapshot_download to ensure ALL files (tokenizer, config, weights) are present
        try:
            snapshot_download(
                repo_id=self.MODEL_NAME,
                local_dir=str(local_model_dir),
                token=settings.hf_token,
            )
        except Exception as dl_exc:
            log(f"Snapshot Download Failed: {dl_exc}", level="error")
            # If download fails, we try loading directly from Hub as last resort
            return _create(self.MODEL_NAME, device=target_device)

        # Retry loading from local after download
        try:
            return _create(str(local_model_dir), device=target_device)
        except Exception as final_exc:
            log(f"Final Load Failed: {final_exc}", level="error")
            # Last resort: try Hub direct load
            return _create(self.MODEL_NAME, device=target_device)

    def encoder_to_cpu(self) -> None:
        """Moves the encoder model to CPU memory.

        Used to free up GPU VRAM for Ollama or other heavy processes when
        embedding operations are idle.
        """
        if hasattr(self, "encoder") and self.encoder is not None:
            try:
                self.encoder = self.encoder.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                log("Encoder moved to CPU, VRAM freed for Ollama")
            except Exception as e:
                log(f"Failed to move encoder to CPU: {e}", level="WARNING")

    def encoder_to_gpu(self) -> None:
        """Moves the encoder model back to the configured GPU device.

        Enables high-performance embedding operations. Fallback to CPU if
        GPU move fails.
        """
        device = settings.device or "cuda"
        if (
            device != "cpu"
            and hasattr(self, "encoder")
            and self.encoder is not None
        ):
            try:
                self.encoder = self.encoder.to(device)
                log(f"Encoder moved back to {device}")
            except Exception as e:
                log(f"Failed to move encoder to GPU: {e}", level="WARNING")

    async def _ensure_encoder_loaded(self, job_id: str | None = None) -> None:
        """Initializes the encoder model if it hasn't been loaded yet.

        Updates the last used timestamp for idle unloading logic.
        Uses RESOURCE_ARBITER to manage VRAM.
        """
        if self.encoder is None:
            # Determine VRAM requirement
            vram_gb = 1.0  # Default for small models
            model_lower = self.MODEL_NAME.lower()
            if "nv-embed-v2" in model_lower:
                vram_gb = 16.0
            elif "sfr-embedding-2" in model_lower:
                vram_gb = 4.0
            elif "bge-m3" in model_lower:
                vram_gb = 2.0

            from core.utils.resource_arbiter import RESOURCE_ARBITER

            log(
                f"Lazy loading encoder: {self.MODEL_NAME} ({vram_gb}GB VRAM requested)..."
            )

            async with RESOURCE_ARBITER.acquire(
                "embedding_encoder", vram_gb=vram_gb, job_id=job_id
            ):
                self.encoder = self._load_encoder()

        self._encoder_last_used = time.time()

    def unload_encoder_if_idle(self) -> bool:
        """Unloads the encoder model from memory if it has exceeded idle time.

        Should be called periodically (e.g., in a background task) to optimize
        VRAM usage.

        Returns:
            True if the model was unloaded, False otherwise.
        """
        if self.encoder is None:
            return False
        idle_time = time.time() - self._encoder_last_used
        if idle_time > self._idle_unload_seconds:
            log(f"Unloading encoder after {idle_time:.0f}s idle")
            self.encoder = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()
            return True
        return False

    @observe("db_encode_texts")
    async def encode_texts(
        self,
        texts: str | list[str],
        batch_size: int = 1,
        show_progress_bar: bool = False,
        is_query: bool = False,
        job_id: str | None = None,
    ) -> list[list[float]]:
        """Transforms text or a list of texts into vector embeddings.

        Handles lazy model loading, instructs models (e5, nv-embed) with
        required prefixes, and performs batch encoding.

        Args:
            texts: Single string or list of strings to encode.
            batch_size: Number of texts to process simultaneously.
            show_progress_bar: passed to SentenceTransformer.encode.
            is_query: Whether these texts are search queries (affects prefixes).

        Returns:
            A list of embedding vectors (list of floats).
        """
        # LAZY LOAD on first use
        await self._ensure_encoder_loaded(job_id=job_id)

        # Prepare texts
        if isinstance(texts, str):
            input_texts = [texts]
        else:
            input_texts = list(texts)

        # Check Cache
        cached_embeddings = []
        indices_to_compute = []

        # Keys involve text + is_query + model_name to be safe
        cache_keys = [(t, is_query, self.MODEL_NAME) for t in input_texts]

        results = [None] * len(input_texts)

        for i, key in enumerate(cache_keys):
            if key in self._embedding_cache:
                # Cache hit - move to end (LRU behavior)
                results[i] = self._embedding_cache.pop(key)
                self._embedding_cache[key] = results[i]
            else:
                indices_to_compute.append(i)

        if not indices_to_compute:
            return [list(r) for r in results]

        # Prepare texts for computing
        texts_to_compute = [input_texts[i] for i in indices_to_compute]

        # e5 models require prefix (query: or passage:)
        model_lower = self.MODEL_NAME.lower()
        processed_texts = texts_to_compute

        if "e5" in model_lower:
            prefix = "query: " if is_query else "passage: "
            processed_texts = [prefix + t for t in texts_to_compute]
        elif "nv-embed-v2" in model_lower:
            # NV-Embed-v2 instructions
            if is_query:
                prefix = "Instruction: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
                processed_texts = [prefix + t for t in texts_to_compute]
            # No prefix for passages
        elif "mxbai" in model_lower:
            # mxbai-embed-large-v1 instructions
            if is_query:
                prefix = (
                    "Represent this sentence for searching relevant passages: "
                )
                processed_texts = [prefix + t for t in texts_to_compute]
            # No prefix for passages

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if self.encoder is None:
            # Lazy load if needed or raise error
            await self._ensure_encoder_loaded(job_id=job_id)
            if self.encoder is None:
                log("Encoder not available", level="ERROR")
                return []

        # Encode uncached texts
        embeddings = self.encoder.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        # Update results and cache
        computed_list = [list(e) for e in embeddings]
        for idx_in_batch, original_idx in enumerate(indices_to_compute):
            emb = computed_list[idx_in_batch]
            results[original_idx] = emb

            # Add to cache
            key = cache_keys[original_idx]
            self._embedding_cache[key] = emb

            # Simple eviction
            if len(self._embedding_cache) > self._embedding_cache_max_size:
                # Remove first element (LRU)
                try:
                    self._embedding_cache.pop(next(iter(self._embedding_cache)))
                except Exception:
                    pass

        return results

    async def get_embedding(self, text: str) -> list[float]:
        """Convenience method to get a single query embedding for a string.

        Args:
            text: The text string to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        try:
            return (await self.encode_texts(text, is_query=True))[0]
        except Exception as e:
            log(
                f"Error generating embedding for '{text[:20]}...': {e}",
                level="ERROR",
            )
            # Return zero vector as fallback to prevent crash
            return [0.0] * self.MEDIA_VECTOR_SIZE

    async def encode_text(self, text: str) -> list[float]:
        """Encodes a single text string."""
        return (await self.encode_texts(text, is_query=True))[0]

    def insert_masklet(
        self,
        media_path: str,
        concept: str,
        start_time: float,
        end_time: float,
        confidence: float = 1.0,
        payload: dict | None = None,
    ) -> None:
        """Inserts a visual grounding masklet into the database.

        Args:
            media_path: Path to the media file.
            concept: Textual description of the concept (e.g. "red car").
            start_time: Start timestamp of the appearance.
            end_time: End timestamp.
            confidence: Detection confidence.
            payload: Additional metadata (bbox, frame_idx, resolution).
        """
        import uuid

        point_id = str(uuid.uuid4())
        full_payload = {
            "media_path": media_path,
            "concept": concept,
            "start": start_time,
            "end": end_time,
            "confidence": confidence,
            "type": "masklet",
            **(payload or {}),
        }

        # Ensure collection exists (lazy check)
        # In prod, this should be in ensure_collections
        collection_name = "media_masklets"

        # Sanitize payload
        full_payload = sanitize_numpy_types(full_payload)

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=[],  # Masklets might not have vectors initially, or query vector of concept?
                        # Ideally we embed the concept text so we can search masklets by similarity
                        # But for now, just storage.
                        # Using empty vector requires configuration that allows it, or dummy vector.
                        # Let's assume we don't index by vector yet, just storage.
                        # WAIT: Qdrant points need vectors if collection is configured with them.
                        # If we assume 'media_masklets' doesn't exist, we might need to create it.
                        payload=full_payload,
                    )
                ],
            )
        except Exception as e:
            # If collection doesn't exist, log warning or try to create?
            # For now, just log to avoid breaking the pipeline if this feature is experimental
            log(f"Failed to insert masklet (collection {collection_name} might be missing): {e}", level="WARNING")

    def extract_concepts_from_video(self, video_path: str, limit: int = 20) -> list[str]:
        """Extracts top frequent concepts/entities from a video's indexed metadata.

        Used to seed the grounding pipeline if no explicit concepts are provided.
        Scans frames for detected objects (if available in payload).
        """
        # Placeholder implementation - we need to see what's actually in frame payloads.
        # Assuming we have 'objects' or 'ocr_text' in frame payloads.
        
        # 1. Scroll frames for this video
        try:
            frames = self.get_frames_for_video(video_path)
            
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

    def get_frames_for_video(self, video_path: str) -> list[dict]:
        """Retrieves all indexed frame metadata for a specific video path.

        Args:
            video_path: The exact path string of the target video.

        Returns:
            A list of payload dictionaries containing frame metadata.
        """
        frames = []
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
                scroll_filter=build_filter([media_path_filter(video_path)]),
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                frames.append(point.payload)
            if offset is None:
                break
        return frames

    def get_frames_by_video(
        self,
        video_path: str,
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

        # Build filter conditions
        must_conditions = [
            models.FieldCondition(
                key="video_path",
                match=models.MatchValue(value=video_path),
            )
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
        video_path: str,
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
            log(f"Failed to get loudness events: {e}")

        events.sort(key=lambda x: x.get("timestamp", 0))
        return events

    def get_voice_segments_by_video(
        self,
        video_path: str,
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
                collection_name=self.VOICE_COLLECTION,
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

    def _check_and_fix_collection(
        self,
        collection_name: str,
        expected_size: int,
        distance: models.Distance = models.Distance.COSINE,
        is_multi_vector: bool = False,
        multi_vector_config: dict | None = None,
    ) -> None:
        """Checks if a collection exists and has the correct dimension. Recreates if mismatch."""
        if self.client.collection_exists(collection_name):
            try:
                info = self.client.get_collection(collection_name)
                vectors_config = info.config.params.vectors

                # Check if we need to check specific multi-vector keys
                if is_multi_vector and isinstance(vectors_config, dict):
                    # Deep check for multi-vector keys
                    if multi_vector_config:
                        for key, expected_params in multi_vector_config.items():
                            if key in vectors_config:
                                existing_size = vectors_config[key].size
                                expected_dim = expected_params.size
                                if existing_size != expected_dim:
                                    log(
                                        f"Collection {collection_name} vector '{key}' mismatch: {existing_size} vs {expected_dim}. Recreating.",
                                        level="WARNING",
                                    )
                                    self.client.delete_collection(
                                        collection_name
                                    )
                                    break  # Break to recreate
                            else:
                                # Start fresh if key missing (optional, but cleaner for schema evolution)
                                log(
                                    f"Collection {collection_name} missing vector '{key}'. Recreating.",
                                    level="WARNING",
                                )
                                self.client.delete_collection(collection_name)
                                break
                elif not is_multi_vector and isinstance(vectors_config, dict):
                    # Mismatch: expected single, got multi
                    log(
                        f"Collection {collection_name} is multi-vector but expected single. Recreating.",
                        level="WARNING",
                    )
                    self.client.delete_collection(collection_name)
                elif hasattr(vectors_config, "size"):
                    existing_size = vectors_config.size
                    if existing_size != expected_size:
                        log(
                            f"Collection {collection_name} dimension mismatch: {existing_size} vs {expected_size}. Recreating.",
                            level="WARNING",
                        )
                        self.client.delete_collection(collection_name)
            except Exception as e:
                log(
                    f"Failed to check {collection_name} dimension: {e}",
                    level="ERROR",
                )

        # Create if missing (or deleted above)
        if not self.client.collection_exists(collection_name):
            if is_multi_vector and multi_vector_config:
                vectors_config = multi_vector_config
            else:
                vectors_config = models.VectorParams(
                    size=expected_size,
                    distance=distance,
                )

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
            log(f"Created collection {collection_name} with valid config.")

    def _ensure_collections(self) -> None:
        # 1. Media Segments (Text Only)
        self._check_and_fix_collection(
            self.MEDIA_SEGMENTS_COLLECTION, self.TEXT_DIM
        )

        # 2. Media Frames (Visual + Metadata)
        self._check_and_fix_collection(
            self.MEDIA_COLLECTION, self.MEDIA_VECTOR_SIZE
        )
        # Re-create indexes for Frames if they were wiped
        self.client.create_payload_index(
            collection_name=self.MEDIA_COLLECTION,
            field_name="video_path",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self._create_text_index(
            collection_name=self.MEDIA_COLLECTION,
            field_name="transcript",
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
        )
        self.client.create_payload_index(
            collection_name=self.MEDIA_COLLECTION,
            field_name="scan_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self._create_text_index(
            collection_name=self.MEDIA_COLLECTION,
            field_name="ocr_text",
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        )

        # 3. Faces (Euclidean Distance)
        self._check_and_fix_collection(
            self.FACES_COLLECTION,
            self.FACE_VECTOR_SIZE,
            distance=models.Distance.EUCLID,
        )

        # 4. Scenelets (Multi-Vector-ish: "content")
        # Scenelets store TEXT SUMMARIES ("content"), so they must use TEXT_DIM (1024 for BGE)
        # NOT Media Vector Size (1152 for visual).
        self._check_and_fix_collection(
            self.SCENELETS_COLLECTION,
            self.TEXT_DIM,
            is_multi_vector=True,
            multi_vector_config={
                "content": models.VectorParams(
                    size=self.TEXT_DIM,
                    distance=models.Distance.COSINE,
                ),
            },
        )
        self.client.create_payload_index(
            collection_name=self.SCENELETS_COLLECTION,
            field_name="media_path",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.SCENELETS_COLLECTION,
            field_name="start_time",
            field_schema=models.PayloadSchemaType.FLOAT,
        )

        # 5. Voice Segments
        self._check_and_fix_collection(
            self.VOICE_COLLECTION, self.VOICE_VECTOR_SIZE
        )

        # 6. Scenes (CRITICAL: Fix Dimension Mismatch)
        # 1024d for BGE-M3, 768d for others.
        # This was missing, causing 768d schema to persist even when model changed.
        visual_features_dim = getattr(
            settings, "visual_features_dim", 1152
        )  # SigLIP default
        video_embedding_dim = getattr(
            settings, "video_embedding_dim", 1024
        )  # InternVideo/LanguageBind

        self._check_and_fix_collection(
            self.SCENES_COLLECTION,
            self.MEDIA_VECTOR_SIZE,  # This is the dynamic one (1024 or 768)
            is_multi_vector=True,
            multi_vector_config={
                "visual": models.VectorParams(
                    size=self.TEXT_DIM,
                    distance=models.Distance.COSINE,
                ),
                "motion": models.VectorParams(
                    size=self.TEXT_DIM,
                    distance=models.Distance.COSINE,
                ),
                "dialogue": models.VectorParams(
                    size=self.TEXT_DIM,
                    distance=models.Distance.COSINE,
                ),
                "visual_features": models.VectorParams(
                    size=visual_features_dim,
                    distance=models.Distance.COSINE,
                ),
                "internvideo": models.VectorParams(
                    size=video_embedding_dim,
                    distance=models.Distance.COSINE,
                ),
                "languagebind": models.VectorParams(
                    size=video_embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            },
        )

        # 8. Summaries
        self._check_and_fix_collection(
            self.SUMMARIES_COLLECTION, self.MEDIA_VECTOR_SIZE
        )

        # Create Payload Indexes for Keyword Search (Full Text)
        # This is CRITICAL for hybrid search to work without scanning the entire DB in memory
        text_fields = [
            "action",
            "dialogue",
            "description",
            "entities",
            "visible_text",
            "face_names",
            "clothing_colors",
            "clothing_types",
            "accessories",
            "scene_location",
            "scene_type",
        ]
        for field in text_fields:
            try:
                self._create_text_index(
                    collection_name=self.MEDIA_COLLECTION,
                    field_name=field,
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    lowercase=True,
                )
            except Exception as e:
                # Index might already exist
                log(
                    f"Index creation for {field} failed (likely exists): {e}",
                    level="DEBUG",
                )

        # Audio Events Collection (CLAP embeddings for semantic audio search)
        # CLAP produces 512-dim embeddings for audio-text similarity
        CLAP_DIM = 512
        if not self.client.collection_exists(self.AUDIO_EVENTS_COLLECTION):
            self.client.create_collection(
                collection_name=self.AUDIO_EVENTS_COLLECTION,
                vectors_config=models.VectorParams(
                    size=CLAP_DIM,  # CLAP embedding dimension
                    distance=models.Distance.COSINE,
                ),
            )
            self.client.create_payload_index(
                collection_name=self.AUDIO_EVENTS_COLLECTION,
                field_name="media_path",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        # Video Metadata Collection
        if not self.client.collection_exists(self.VIDEO_METADATA_COLLECTION):
            self.client.create_collection(
                collection_name=self.VIDEO_METADATA_COLLECTION,
                vectors_config=models.VectorParams(
                    size=1,
                    distance=models.Distance.COSINE,
                ),
            )
            self.client.create_payload_index(
                collection_name=self.VIDEO_METADATA_COLLECTION,
                field_name="media_path",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        # Masklets Collection (Video Concept Segmentation)
        self._check_and_fix_collection(
            self.MASKLETS_COLLECTION,
            self.TEXT_DIM,  # Assuming text-based concept search, or visual? If MaskLet is concept TEXT, use TEXT_DIM
            # Wait, if masklets are tracked objects, they might be visual features?
            # Based on "Failed to fetch masklets", let's assume standard TEXT_DIM for now until proven otherwise.
            # Actually, if it's "concept", it's likely text embedding of the concept name.
        )
        self.client.create_payload_index(
            collection_name=self.MASKLETS_COLLECTION,
            field_name="media_path",  # Standardized to media_path (was video_path)
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        log("Qdrant collections and indexes ensured")

    def _create_text_index(
        self,
        collection_name: str,
        field_name: str,
        type: models.TextIndexType = models.TextIndexType.TEXT,
        tokenizer: models.TokenizerType = models.TokenizerType.WORD,
        min_token_len: int = 2,
        max_token_len: int = 20,
        lowercase: bool = True,
    ) -> None:
        """Helper to create a text payload index with common settings."""
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.TextIndexParams(
                type=type,
                tokenizer=tokenizer,
                min_token_len=min_token_len,
                max_token_len=max_token_len,
                lowercase=lowercase,
            ),
        )

    def list_collections(self) -> models.CollectionsResponse:
        """List all collections in the Qdrant instance."""
        return self.client.get_collections()

    @observe("db_insert_media_segments")
    async def insert_media_segments(
        self,
        video_path: str,
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
            wait=False,
        )

    @observe("db_search_media")
    async def search_frames(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = None,  # Uses settings.frame_search_score_threshold if None
        allowed_video_paths: list[str] | None = None,
        face_cluster_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for frames using text query.

        Args:
            query: The search text.
            limit: Max results.
            score_threshold: Min similarity score.
            allowed_video_paths: Optional list of video paths to restrict search.
            face_cluster_id: Optional filter for specific face cluster.

        Returns:
            A list of payload dictionaries containing frame metadata.
        """
        # CRITICAL: Use Visual Encoder (SigLIP/CLIP) to encode text for Frame Search
        # Frames are indexed with Visual Encoder, so query MUST be in same latent space.
        # BGE is for TEXT-only collections - using it here would cause ~0 similarity.
        try:
            query_vector = await self.visual_encoder.encode_text(query)
            if not query_vector:
                log(
                    f"Visual encoder returned empty for query: '{query[:50]}'",
                    level="WARNING",
                )
                return []  # Return empty, not garbage results
        except Exception as e:
            log(
                f"Visual Text Encoding failed: {e}. Cannot search frames without visual encoder.",
                level="ERROR",
            )
            return []  # Return empty, not wrong-space results

        # Build filter conditions
        filter_conditions = []
        if allowed_video_paths:
            filter_conditions.append(
                models.FieldCondition(
                    key="video_path",
                    match=models.MatchAny(any=allowed_video_paths),
                )
            )

        if face_cluster_id is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="face_cluster_ids",
                    match=models.MatchValue(value=face_cluster_id),
                )
            )

        scroll_filter = None
        if filter_conditions:
            scroll_filter = models.Filter(
                must=cast(list[models.Condition], filter_conditions)
            )

        results = self.client.query_points(
            collection_name=self.MEDIA_COLLECTION,
            query=query_vector,
            query_filter=scroll_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        ).points
        return [point.payload for point in results if point.payload]

    @observe("db_search_media")
    async def search_media(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
        video_path: str | None = None,
        segment_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Searches for media segments (dialogue/subtitles) similar to the query.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score (0.0 to 1.0).
            video_path: If provided, restricts search to this specific video.
            segment_type: Filter by 'dialogue', 'subtitle', etc.

        Returns:
            A list of result dictionaries containing score, text, and timestamps.
        """
        query_vector = (await self.encode_texts(query, is_query=True))[0]

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
    @retry_on_connection_error()
    def upsert_media_frame(
        self,
        point_id: str,
        vector: list[float],
        video_path: str,
        timestamp: float,
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
            end_time = frame.get("end_time") or ts + settings.search_default_duration
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

    async def search_scenelets(
        self,
        query: str,
        limit: int = 10,
        video_path: str | None = None,
        gap_threshold: float = 2.0,
        padding: float = 3.0,
    ) -> list[dict]:
        """Search for dynamic video segments based on dense frame retrieval.

        Performs semantic search on frames, then clusters temporally adjacent
        matches to form coherent 'scenelets'. Applies padding to capture context.

        Args:
            query: Natural language search query.
            limit: Number of final scenelets to return.
            video_path: Optional filter for specific video.
            gap_threshold: Max seconds between frames to merge into one cluster.
            padding: Seconds to add before start and after end.

        Returns:
            List of dicts with 'start_time', 'end_time', 'score', etc.
        """
        # 1. Retrieve raw frame candidates
        # We fetch more than limit because clusters will reduce count
        raw_limit = limit * 5

        # Reuse existing frame search logic but get raw points
        # FIX: Use Visual Encoder for Scenelets (based on Frames)
        try:
            query_vector = await self.visual_encoder.encode_text(query)
            if not query_vector:
                query_vector = (await self.encode_texts(query, is_query=True))[
                    0
                ]
        except Exception:
            query_vector = (await self.encode_texts(query, is_query=True))[0]

        filters = []
        if video_path:
            filters.append(
                models.FieldCondition(
                    key="video_path",
                    match=models.MatchValue(value=video_path),
                )
            )

        try:
            # FIX: Use query_points instead of search (deprecated/wrong method name for Client)
            resp = self.client.query_points(
                collection_name=self.MEDIA_COLLECTION,
                query=query_vector,
                limit=raw_limit,
                query_filter=models.Filter(must=filters) if filters else None,
                with_payload=True,
            )
            hits = resp.points
        except Exception as e:
            log(f"[Scenelet] Frame search failed: {e}", level="ERROR")
            return []

        if not hits:
            return []

        # 2. Group by Video
        # (Though usually we search one video or global, let's handle mixed)
        hits_by_video = {}
        for hit in hits:
            payload = hit.payload or {}
            v_path = payload.get("video_path")
            if not v_path:
                continue

            if v_path not in hits_by_video:
                hits_by_video[v_path] = []

            # Extract timestamp
            ts = payload.get("timestamp")
            if ts is None:
                # Try start_time alias
                ts = payload.get("start_time", 0.0)

            hits_by_video[v_path].append(
                {
                    "score": hit.score,
                    "timestamp": float(ts),
                    "text": payload.get("description", "")
                    or payload.get("ocr_text", "")
                    or "",
                    "payload": payload,
                }
            )

        final_scenelets = []

        # 3. Clustering & Padding per Video
        for v_path, candidates in hits_by_video.items():
            # Sort by timestamp
            candidates.sort(key=lambda x: x["timestamp"])

            clusters = []
            if not candidates:
                continue

            # Current cluster state
            current_cluster = [candidates[0]]

            for i in range(1, len(candidates)):
                curr = candidates[i]
                prev = candidates[i - 1]

                # Check time gap
                if (curr["timestamp"] - prev["timestamp"]) <= gap_threshold:
                    current_cluster.append(curr)
                else:
                    # Finalize current cluster
                    clusters.append(current_cluster)
                    current_cluster = [curr]

            # Append last cluster
            if current_cluster:
                clusters.append(current_cluster)

            # Process Clusters into Scenelets
            for cl in clusters:
                # Core bounds
                start_ts = cl[0]["timestamp"]
                end_ts = cl[-1]["timestamp"]

                # Scores
                max_score = max(c["score"] for c in cl)
                avg_score = sum(c["score"] for c in cl) / len(cl)

                # Descriptions (Best score's desc)
                best_frame = max(cl, key=lambda x: x["score"])
                description = best_frame["text"]

                # Pad
                final_start = max(0.0, start_ts - padding)
                # Note: We need video duration to clamp end efficiently.
                # For now, we rely on UI to handle over-bounds or just let it be.
                # Or check if metadata exists. A simple approach is just setting it.
                final_end = end_ts + padding

                final_scenelets.append(
                    {
                        "video_path": v_path,
                        "start_time": final_start,
                        "end_time": final_end,
                        "core_start": start_ts,
                        "core_end": end_ts,
                        "score": max_score,  # Use max for retrieval ranking
                        "text": description,
                        "frame_count": len(cl),
                        "best_frame_timestamp": best_frame["timestamp"],
                    }
                )

        # 4. Sort and Limit
        final_scenelets.sort(key=lambda x: x["score"], reverse=True)
        return final_scenelets[:limit]

    @observe("db_insert_masklet")
    def insert_masklet(
        self,
        video_path: str,
        concept: str,
        start_time: float,
        end_time: float,
        confidence: float = 1.0,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Inserts a masklet (video segment tracking a specific concept).

        Args:
            video_path: Path to the source video file.
            concept: The concept name/label being tracked.
            start_time: Start timestamp of the masklet in seconds.
            end_time: End timestamp of the masklet in seconds.
            confidence: Confidence score of the tracking.
            payload: Optional additional metadata for the masklet.
        """
        unique_str = f"{video_path}_{concept}_{start_time}_{end_time}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

        final_payload = {
            "video_path": video_path,
            "concept": concept,
            "start_time": start_time,
            "end_time": end_time,
            "confidence": confidence,
            "type": "masklet",
        }
        if payload:
            final_payload.update(payload)

        try:
            self.client.upsert(
                collection_name=self.MASKLETS_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=[1.0],  # Dummy vector, as we filter by payload
                        payload=final_payload,
                    )
                ],
                wait=False,
            )
        except Exception as e:
            log(f"Failed to insert masklet: {e}", level="ERROR")

    @observe("db_update_masklet")
    def update_masklet(
        self,
        masklet_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Updates an existing masklet payload.

        Args:
            masklet_id: The ID of the masklet point.
            updates: Dictionary of fields to update in the payload.

        Returns:
            True if successful.
        """
        try:
            # We use set_payload to update specific fields without rewriting the whole point
            self.client.set_payload(
                collection_name=self.MASKLETS_COLLECTION,
                payload=updates,
                points=[masklet_id],
            )
            return True
        except Exception as e:
            log(f"Failed to update masklet {masklet_id}: {e}", level="ERROR")
            return False

    def get_masklets(
        self,
        video_path: str,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieves masklets for a specific video and optional time range.

        Args:
            video_path: Path to the source video.
            start_time: Optional start search bound.
            end_time: Optional end search bound.

        Returns:
            List of masklet payloads.
        """
        must_filters = [
            models.FieldCondition(
                key="video_path", match=models.MatchValue(value=video_path)
            ),
            models.FieldCondition(
                key="type", match=models.MatchValue(value="masklet")
            ),
        ]

        if start_time is not None:
            must_filters.append(
                models.FieldCondition(
                    key="start_time", range=models.Range(gte=start_time)
                )
            )
        if end_time is not None:
            must_filters.append(
                models.FieldCondition(
                    key="end_time", range=models.Range(lte=end_time)
                )
            )

        try:
            results = self.client.scroll(
                collection_name=self.MASKLETS_COLLECTION,
                scroll_filter=models.Filter(
                    must=cast(list[models.Condition], must_filters)
                ),
                limit=1000,
            )
            # Ensure payloads are not None before returning
            valid_payloads: list[dict[str, Any]] = [
                p.payload for p in results[0] if p.payload is not None
            ]
            return valid_payloads
        except Exception as e:
            log(f"Failed to fetch masklets: {e}", level="ERROR")
            return []

    def get_global_summary(self, video_path: str) -> dict[str, Any] | None:
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

    async def search_masklets(
        self,
        query: str,
        limit: int = 10,
        video_path: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Search masklets (tracked objects/concepts) semantically.

        Enables queries like "track the red car" or "follow the person in blue".

        Args:
            query: Natural language search query.
            limit: Maximum results to return.
            video_path: Optional filter by video.
            score_threshold: Minimum similarity score.

        Returns:
            List of matching masklet dictionaries with scores.
        """
        query_vector = await self.encode_texts(query)

        filters = [media_path_filter(video_path)]
        
        try:
            results = self.client.search(
                collection_name=self.MASKLETS_COLLECTION,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=build_filter(filters),
            )

            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "video_path": r.payload.get("media_path", ""),
                    "concept": r.payload.get("concept", ""),
                    "start_time": r.payload.get("start_time", 0),
                    "end_time": r.payload.get("end_time", 0),
                    "confidence": r.payload.get("confidence", 0),
                    **r.payload,
                }
                for r in results
                if r.payload
            ]
        except Exception as e:
            log(f"Masklet search failed: {e}")
            return []

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
                collection_name=self.VOICE_COLLECTION,
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

    def upsert_face_cluster_centroid(
        self, cluster_id: int, embedding: list[float]
    ) -> None:
        """Stores or updates the centroid for a face cluster.

        Uses a deterministic ID based on cluster_id to allow easy retrieval/update.
        """
        import uuid

        # Deterministic UUID for the centroid
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"face_centroid_{cluster_id}"))

        try:
            self.client.upsert(
                collection_name=self.FACE_COLLECTION,
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
            log(f"Failed to upsert face centroid {cluster_id}: {e}", level="ERROR")

    def upsert_voice_cluster_centroid(
        self, cluster_id: int, embedding: list[float]
    ) -> None:
        """Stores or updates the centroid for a voice cluster.

        Uses a deterministic ID based on cluster_id to allow easy retrieval/update.
        """
        import uuid

        # Deterministic UUID for the centroid
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"voice_centroid_{cluster_id}"))

        try:
            self.client.upsert(
                collection_name=self.VOICE_COLLECTION,
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
            log(f"Failed to upsert voice centroid {cluster_id}: {e}", level="ERROR")

    def upsert_speaker_embedding(
        self,
        speaker_id: str,
        embedding: list[float],
        media_path: str,
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
            collection_name=self.VOICE_COLLECTION,
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

    @observe("db_search_frames_filtered")
    async def search_frames_filtered(
        self,
        query_vector: list[float] | str,
        face_cluster_ids: list[int] | None = None,
        limit: int = 20,
        score_threshold: float | None = None,
        video_path: str
        | None = None,  # CRITICAL: Prevent cross-video identity leakage
    ) -> list[dict[str, Any]]:
        """Search frames with optional identity and video filtering.

        Used by agentic search to filter by face_cluster_ids.
        IMPORTANT: Always pass video_path to prevent cross-video identity leakage.

        Args:
            query_vector: The query embedding vector OR natural language query string.
            face_cluster_ids: Face cluster IDs to filter by (identity filter).
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.
            video_path: Filter results to this video only (prevents cross-video leakage).

        Returns:
            A list of matching frames with full payload.
        """
        # Auto-encode if string passed
        if isinstance(query_vector, str):
            query_vector = await self.get_embedding(query_vector)

        # Build filter conditions
        conditions: list[models.Condition] = []

        # CRITICAL: Add video_path filter to prevent cross-video identity leakage
        if video_path:
            conditions.append(
                models.FieldCondition(
                    key="video_path",
                    match=models.MatchValue(value=video_path),
                )
            )

        # Face identity filter
        if face_cluster_ids:
            conditions.append(
                models.FieldCondition(
                    key="face_cluster_ids",
                    match=models.MatchAny(any=face_cluster_ids),
                )
            )

        query_filter = models.Filter(must=conditions) if conditions else None

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
    async def search_frames_hybrid(
        self,
        query: str,
        limit: int = 20,
        video_paths: str | list[str] | None = None,
        face_cluster_ids: list[int] | None = None,
        rrf_k: int = None,  # Uses settings.rrf_constant if None
        transcript_query: str | None = None,
        music_section: str | None = None,
        high_energy: bool = False,
    ) -> list[dict[str, Any]]:
        """Performs a hybrid search combining vector, keyword, and identity filters."""

        from collections import defaultdict

        results_by_id: dict[str, dict] = {}
        rank_lists: dict[str, dict[str, int]] = defaultdict(dict)

        # Normalize video_paths to list
        if isinstance(video_paths, str):
            video_paths = [video_paths]

        # === 1. VECTOR SEARCH (Semantic Understanding) ===
        try:
            await self._ensure_encoder_loaded()
            query_vector = (await self.encode_texts(query, is_query=True))[0]

            conditions = []
            if video_paths:
                conditions.append(
                    models.FieldCondition(
                        key="video_path",
                        match=models.MatchAny(any=video_paths),
                    )
                )
            if face_cluster_ids:
                conditions.append(
                    models.FieldCondition(
                        key="face_cluster_ids",
                        match=models.MatchAny(any=face_cluster_ids),
                    )
                )

            qfilter = models.Filter(must=conditions) if conditions else None

            vec_resp = self.client.query_points(
                collection_name=self.MEDIA_COLLECTION,
                query=query_vector,
                limit=limit * 2,
                query_filter=qfilter,
            )

            for rank, hit in enumerate(vec_resp.points):
                point_id = str(hit.id)
                rank_lists["vector"][point_id] = rank + 1
                if point_id not in results_by_id:
                    payload = hit.payload or {}
                    results_by_id[point_id] = {
                        "id": point_id,
                        "score": 0.0,
                        "vector_score": hit.score,
                        "match_reasons": [],
                        **payload,
                    }
                # Generate detailed match reason with actual content
                payload = hit.payload or {}
                desc_preview = (
                    payload.get("description") or payload.get("action") or ""
                )[:80]
                results_by_id[point_id]["match_reasons"].append(
                    f"Semantic match (score={hit.score:.2f}): {desc_preview}..."
                )
        except Exception as e:
            log(f"Vector search failed: {e}")

        log(f"Vector search found {len(results_by_id)} candidates so far")

        # === 2. KEYWORD SEARCH (Text Fields via Qdrant Indexes) ===
        try:
            # We use Qdrant's MatchText to find documents containing query terms.
            # This is much faster than scraping random frames.

            # Fields to search - COMPREHENSIVE list of ALL indexed text fields
            text_fields = [
                "action",
                "dialogue",
                "description",
                "entities",
                "visible_text",
                "face_names",
                "speaker_names",
                "visual_attributes",  # Dynamic: ALL visual details from VLM
                "entity_details",  # Dynamic: ALL entity names and categories
                "scene_location",
                "ocr_text",
                "transcript",
                "identity_text",
                "temporal_context",
            ]

            should_conditions = []
            for field in text_fields:
                should_conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchText(text=query)
                    )
                )

            # Combine with strong filters (video path)
            must_conditions = []
            if video_paths:
                must_conditions.append(
                    models.FieldCondition(
                        key="video_path",
                        match=models.MatchAny(any=video_paths),
                    )
                )

            keyword_filter = models.Filter(
                should=should_conditions,
                must=must_conditions if must_conditions else None,
            )

            # Scroll for matches (limit to 100 high-relevance matches)
            # Since Qdrant basic text match doesn't score, we treat them as high-confidence hits.
            # Ideally we'd use sparse vectors for BM25, but this is a robust fallback.
            scroll_resp = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
                scroll_filter=keyword_filter,
                limit=limit * 3,
                with_payload=True,
                with_vectors=False,
            )

            for rank, point in enumerate(scroll_resp[0]):
                point_id = str(point.id)
                rank_lists["keyword"][point_id] = rank + 1  # 1-based rank

                if point_id not in results_by_id:
                    payload = point.payload or {}
                    results_by_id[point_id] = {
                        "id": point_id,
                        "score": 0.0,
                        "keyword_score": 1.0,  # Placeholder info
                        "match_reasons": [],
                        **payload,
                    }

                # Try to determine WHY it matched for the UI
                payload = results_by_id[point_id]
                matched_fields = []
                q_lower = query.lower().split()

                # Heuristic verify
                for field in text_fields:
                    val = str(payload.get(field, "")).lower()
                    if any(w in val for w in q_lower if len(w) > 2):
                        matched_fields.append(field)

                # Generate detailed match reason showing WHAT matched
                field_details = []
                for field in matched_fields:
                    val = str(payload.get(field, ""))[:50]
                    if val:
                        field_details.append(f"{field}='{val}'")

                if field_details:
                    results_by_id[point_id]["match_reasons"].append(
                        f"Text match: {'; '.join(field_details[:3])}"
                    )
                else:
                    results_by_id[point_id]["match_reasons"].append(
                        "Text match (partial)"
                    )

            # === 2b. MASKLET SEARCH (Deep Video Understanding) ===
            # Search for SAM3-tracked concepts overlap with query
            try:
                masklet_conditions = []
                if video_paths:
                    masklet_conditions.append(
                        models.FieldCondition(
                            key="video_path",
                            match=models.MatchAny(any=video_paths),
                        )
                    )
                # Check for concept match
                # Use MatchText for concepts too
                masklet_conditions.append(
                    models.FieldCondition(
                        key="concept",
                        match=models.MatchText(text=query),
                    )
                )

                masklet_filter = models.Filter(must=masklet_conditions)

                mask_resp = self.client.scroll(
                    collection_name=self.MASKLETS_COLLECTION,  # Fixed: use constant
                    scroll_filter=masklet_filter,
                    limit=50,
                    with_payload=True,
                )

                for point in mask_resp[0]:
                    payload = point.payload or {}
                    point_id = str(point.id)  # Use unique ID

                    rank_lists["keyword"][point_id] = 1  # High rank

                    if point_id not in results_by_id:
                        results_by_id[point_id] = {
                            "id": point_id,
                            "score": 0.0,
                            "keyword_score": 0.9,  # High confidence
                            "match_reasons": [],
                            "timestamp": payload.get("start_time", 0.0),
                            "video_path": payload.get("video_path"),
                            "action": f"Tracked concept: {payload.get('concept')}",
                            "type": "masklet",
                        }
                    results_by_id[point_id]["match_reasons"].append(
                        f"Concept match: {payload.get('concept')}"
                    )
            except Exception as e:
                log(f"Masklet search failed: {e}")

        except Exception as e:
            log(f"Keyword search failed: {e}")

        log(f"Keyword search found {len(rank_lists['keyword'])} matches")

        # === 3. IDENTITY SEARCH (Face/Speaker Names) ===
        try:
            identity_names = self._extract_identity_names(query)
            if identity_names:
                for name in identity_names:
                    cluster_id = self.fuzzy_get_cluster_id_by_name(name)
                    if cluster_id is not None:
                        conditions = [
                            models.FieldCondition(
                                key="face_cluster_ids",
                                match=models.MatchAny(any=[cluster_id]),
                            )
                        ]
                        if video_paths:
                            conditions.append(
                                models.FieldCondition(
                                    key="video_path",
                                    match=models.MatchAny(any=video_paths),
                                )
                            )

                        identity_resp = self.client.scroll(
                            collection_name=self.MEDIA_COLLECTION,
                            scroll_filter=models.Filter(must=conditions),  # type: ignore
                            limit=limit * 2,
                            with_payload=True,
                        )

                        for rank, point in enumerate(identity_resp[0]):
                            point_id = str(point.id)
                            rank_lists["identity"][point_id] = rank + 1
                            if point_id not in results_by_id:
                                payload = point.payload or {}
                                results_by_id[point_id] = {
                                    "id": point_id,
                                    "score": 0.0,
                                    "identity_match": True,
                                    "match_reasons": [],
                                    **payload,
                                }
                            # Get face names from the result payload
                            face_names = results_by_id[point_id].get(
                                "face_names", []
                            )
                            confidence = (
                                "high"
                                if len(rank_lists.get("identity", {})) <= 5
                                else "medium"
                            )
                            # Better formatting for unknown faces
                            display_name = name
                            if (
                                not display_name
                                or display_name.lower().startswith("unknown")
                            ):
                                display_name = f"Person {cluster_id}"

                            results_by_id[point_id]["match_reasons"].append(
                                f"Face identity: '{display_name}' (cluster={cluster_id}, conf={confidence}, visible_faces={face_names})"
                            )
                            results_by_id[point_id]["matched_identity"] = (
                                display_name
                            )
        except Exception as e:
            log(f"Identity search failed: {e}")

        # === 4. VOICE/TRANSCRIPT SEARCH (Who said what) ===
        try:
            # Search voice_segments for dialogue matching query
            voice_conditions = []
            if video_paths:
                voice_conditions.append(
                    models.FieldCondition(
                        key="media_path",
                        match=models.MatchAny(any=video_paths),
                    )
                )
            # Text match on transcript
            voice_conditions.append(
                models.FieldCondition(
                    key="transcript",
                    match=models.MatchText(text=transcript_query or query),
                )
            )

            voice_resp = self.client.scroll(
                collection_name=self.VOICE_COLLECTION,
                scroll_filter=models.Filter(must=voice_conditions),
                limit=limit * 2,
                with_payload=True,
            )

            for rank, point in enumerate(voice_resp[0]):
                payload = point.payload or {}
                # Create composite ID linking to timestamp
                media_path = payload.get("media_path", "")
                start_time = payload.get("start", 0)

                # Find corresponding frame at this timestamp
                frame_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_path",
                            match=models.MatchValue(value=media_path),
                        ),
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(
                                gte=start_time - 2, lte=start_time + 2
                            ),
                        ),
                    ]
                )

                frame_resp = self.client.scroll(
                    collection_name=self.MEDIA_COLLECTION,
                    scroll_filter=frame_filter,
                    limit=1,
                    with_payload=True,
                )

                if frame_resp[0]:
                    frame_point = frame_resp[0][0]
                    point_id = str(frame_point.id)
                    rank_lists["voice"][point_id] = rank + 1

                    if point_id not in results_by_id:
                        frame_payload = frame_point.payload or {}
                        results_by_id[point_id] = {
                            "id": point_id,
                            "score": 0.0,
                            "voice_match": True,
                            "match_reasons": [],
                            **frame_payload,
                        }

                    # Detailed voice/transcript match info
                    transcript = payload.get("transcript", "")[:100]
                    speaker = (
                        payload.get("speaker_name")
                        or f"Speaker #{payload.get('cluster_id', '?')}"
                    )
                    results_by_id[point_id]["match_reasons"].append(
                        f"Dialogue match: '{speaker}' said '{transcript}...'"
                    )
                    results_by_id[point_id]["matched_dialogue"] = transcript

        except Exception as e:
            log(f"Voice/transcript search failed: {e}")

        # === 4b. MEDIA SEGMENT SEARCH (ASR/Subtitles) ===
        # Fallback for when voice diarization (VOICE_COLLECTION) misses segments
        try:
            asr_conditions = []
            if video_paths:
                asr_conditions.append(
                    models.FieldCondition(
                        key="video_path",
                        match=models.MatchAny(any=video_paths),
                    )
                )

            asr_conditions.append(
                models.FieldCondition(
                    key="text",
                    match=models.MatchText(text=transcript_query or query),
                )
            )

            asr_resp = self.client.scroll(
                collection_name=self.MEDIA_SEGMENTS_COLLECTION,
                scroll_filter=models.Filter(must=asr_conditions),
                limit=limit * 2,
                with_payload=True,
            )

            for rank, point in enumerate(asr_resp[0]):
                payload = point.payload or {}
                media_path = payload.get("video_path", "")
                start_time = payload.get("start", 0)
                text = payload.get("text", "")[:100]

                # Find matching frame
                frame_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_path",
                            match=models.MatchValue(value=media_path),
                        ),
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(
                                gte=start_time - 2, lte=start_time + 2
                            ),
                        ),
                    ]
                )

                frame_resp = self.client.scroll(
                    collection_name=self.MEDIA_COLLECTION,
                    scroll_filter=frame_filter,
                    limit=1,
                    with_payload=True,
                )

                if frame_resp[0]:
                    frame_point = frame_resp[0][0]
                    point_id = str(frame_point.id)

                    # Merge into voice rank list (treating ASR as voice)
                    if point_id not in rank_lists["voice"]:
                        rank_lists["voice"][point_id] = rank + 1
                    else:
                        rank_lists["voice"][point_id] = min(
                            rank_lists["voice"][point_id], rank + 1
                        )

                    if point_id not in results_by_id:
                        frame_payload = frame_point.payload or {}
                        results_by_id[point_id] = {
                            "id": point_id,
                            "score": 0.0,
                            "voice_match": True,
                            "match_reasons": [],
                            **frame_payload,
                        }

                    if (
                        f"Dialogue match: '{text}...'"
                        not in results_by_id[point_id]["match_reasons"]
                    ):
                        results_by_id[point_id]["match_reasons"].append(
                            f"Transcript match: '{text}...'"
                        )
                        results_by_id[point_id]["matched_dialogue"] = text

        except Exception as e:
            log(f"ASR/Media segment search failed: {e}")

        # === 5. AUDIO EVENT SEARCH (Music/Sounds) ===
        try:
            audio_conditions = []
            if video_paths:
                audio_conditions.append(
                    models.FieldCondition(
                        key="media_path",
                        match=models.MatchAny(any=video_paths),
                    )
                )
            # Match audio event type/label
            audio_conditions.append(
                models.FieldCondition(
                    key="event_type",
                    match=models.MatchText(text=query),
                )
            )

            audio_resp = self.client.scroll(
                collection_name=self.AUDIO_EVENTS_COLLECTION,
                scroll_filter=models.Filter(
                    should=audio_conditions
                ),  # OR condition
                limit=limit,
                with_payload=True,
            )

            for rank, point in enumerate(audio_resp[0]):
                payload = point.payload or {}
                media_path = payload.get("media_path", "")
                start_time = payload.get("start_time", 0)

                # Find frame near this audio event
                if media_path:
                    frame_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="video_path",
                                match=models.MatchValue(value=media_path),
                            ),
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(
                                    gte=start_time - 1, lte=start_time + 3
                                ),
                            ),
                        ]
                    )

                    frame_resp = self.client.scroll(
                        collection_name=self.MEDIA_COLLECTION,
                        scroll_filter=frame_filter,
                        limit=1,
                        with_payload=True,
                    )

                    if frame_resp[0]:
                        frame_point = frame_resp[0][0]
                        point_id = str(frame_point.id)
                        rank_lists["audio"][point_id] = rank + 1

                        if point_id not in results_by_id:
                            frame_payload = frame_point.payload or {}
                            results_by_id[point_id] = {
                                "id": point_id,
                                "score": 0.0,
                                "audio_event_match": True,
                                "match_reasons": [],
                                **frame_payload,
                            }

                        # Detailed audio event info
                        event_type = payload.get("event_type", "unknown")
                        confidence = payload.get("confidence", 0)
                        results_by_id[point_id]["match_reasons"].append(
                            f"Audio event: '{event_type}' (conf={confidence:.2f}) at {start_time:.1f}s"
                        )

        except Exception as e:
            log(f"Audio event search failed: {e}")

        # === 5b. MUSIC SECTION FILTERING (Temporal precision) ===
        # Filter for specific music sections like "chorus" or "drop"
        if music_section:
            try:
                section_event = f"music_{music_section.lower()}"
                section_conditions = [
                    models.FieldCondition(
                        key="event_type",
                        match=models.MatchValue(value=section_event),
                    )
                ]
                if video_paths:
                    section_conditions.append(
                        models.FieldCondition(
                            key="media_path",
                            match=models.MatchAny(any=video_paths),
                        )
                    )

                section_resp = self.client.scroll(
                    collection_name=self.AUDIO_EVENTS_COLLECTION,
                    scroll_filter=models.Filter(must=section_conditions),
                    limit=50,
                    with_payload=True,
                )

                # Get time ranges for this section type
                section_ranges = []
                for point in section_resp[0]:
                    payload = point.payload or {}
                    start = payload.get("start_time", 0)
                    end = payload.get(
                        "end_time", start + 30
                    )  # Default 30s if no end
                    media = payload.get("media_path", "")
                    section_ranges.append((media, start, end))

                if section_ranges:
                    log(
                        f"[Music] Found {len(section_ranges)} '{music_section}' sections"
                    )

                    # Boost results that fall within these time ranges
                    for point_id, result in results_by_id.items():
                        video_path = result.get("video_path", "")
                        timestamp = result.get("timestamp", 0)

                        for media, start, end in section_ranges:
                            if (
                                video_path == media
                                and start <= timestamp <= end
                            ):
                                rank_lists["music_section"][point_id] = (
                                    1  # High rank
                                )
                                result["match_reasons"].append(
                                    f"During {music_section} ({start:.1f}s-{end:.1f}s)"
                                )
                                result["in_music_section"] = music_section
                                break

            except Exception as e:
                log(f"Music section filtering failed: {e}")

        # === 5c. HIGH ENERGY FILTERING ===
        if high_energy:
            try:
                # Filter for high-energy moments (drops, choruses)
                energy_conditions = []
                if video_paths:
                    energy_conditions.append(
                        models.FieldCondition(
                            key="media_path",
                            match=models.MatchAny(any=video_paths),
                        )
                    )

                # Look for drops and choruses (typically high energy)
                energy_resp = self.client.scroll(
                    collection_name=self.AUDIO_EVENTS_COLLECTION,
                    scroll_filter=models.Filter(
                        must=energy_conditions,
                        should=[
                            models.FieldCondition(
                                key="event_type",
                                match=models.MatchValue(value="music_drop"),
                            ),
                            models.FieldCondition(
                                key="event_type",
                                match=models.MatchValue(value="music_chorus"),
                            ),
                        ],
                    )
                    if energy_conditions
                    else models.Filter(
                        should=[
                            models.FieldCondition(
                                key="event_type",
                                match=models.MatchValue(value="music_drop"),
                            ),
                            models.FieldCondition(
                                key="event_type",
                                match=models.MatchValue(value="music_chorus"),
                            ),
                        ]
                    ),
                    limit=50,
                    with_payload=True,
                )

                energy_ranges = []
                for point in energy_resp[0]:
                    payload = point.payload or {}
                    energy_level = payload.get("payload", {}).get("energy", 0)
                    if energy_level > 0.8:  # Only high-energy sections
                        start = payload.get("start_time", 0)
                        end = payload.get("end_time", start + 30)
                        media = payload.get("media_path", "")
                        energy_ranges.append((media, start, end))

                if energy_ranges:
                    log(
                        f"[Energy] Found {len(energy_ranges)} high-energy sections"
                    )

                    for point_id, result in results_by_id.items():
                        video_path = result.get("video_path", "")
                        timestamp = result.get("timestamp", 0)

                        for media, start, end in energy_ranges:
                            if (
                                video_path == media
                                and start <= timestamp <= end
                            ):
                                rank_lists["high_energy"][point_id] = 1
                                result["match_reasons"].append(
                                    f"High-energy moment ({start:.1f}s-{end:.1f}s)"
                                )
                                result["is_high_energy"] = True
                                break

            except Exception as e:
                log(f"High energy filtering failed: {e}")

        log(
            "Total modalities searched: vector, keyword, identity, voice, audio, music_structure"
        )

        # === 6. RRF FUSION ===
        for point_id, result in results_by_id.items():
            rrf_score = 0.0
            for _method, ranks in rank_lists.items():
                if point_id in ranks:
                    rrf_score += 1.0 / (rrf_k + ranks[point_id])
            result["score"] = rrf_score
            result["rrf_score"] = rrf_score

        final_results = sorted(
            results_by_id.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:limit]

        return final_results

    def _extract_identity_names(self, query: str) -> list[str]:
        """Extract potential person names from query for identity search."""
        known_names = set()
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
                limit=500,
                with_payload=["name"],
                with_vectors=False,
            )
            for pt in resp[0]:
                name = (pt.payload or {}).get("name")
                if name:
                    known_names.add(name.lower())
        except Exception as e:
            log(f"extract_names_from_query scroll failed: {e}", level="DEBUG")

        query_lower = query.lower()
        found_names = []
        for name in known_names:
            if name in query_lower:
                found_names.append(name)

        return found_names

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
                collection_name=self.FACES_COLLECTION,
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
        media_path: str,
        event_type: str,
        start_time: float,
        end_time: float,
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

    def update_media_metadata(self, media_path: str, metadata: dict[str, Any]):
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
        except Exception as e:
            log(f"get_face_ids_for_video failed: {e}", level="DEBUG")
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
        if len(embedding) != self.VOICE_VECTOR_SIZE:
            raise ValueError(
                f"voice vector dim mismatch: expected {self.VOICE_VECTOR_SIZE}, "
                f"got {len(embedding)}"
            )

        # Deterministic ID to prevent duplicates (idempotency)
        unique_str = f"voice_{media_path}_{start:.3f}_{end:.3f}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

        payload = {
            "media_path": media_path,
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
            collection_name=self.VOICE_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

    def set_speaker_name(self, cluster_id: int, name: str) -> int:
        """Assign a name to a voice cluster.

        Args:
            cluster_id: The voice cluster ID.
            name: The user-assigned name.

        Returns:
            Number of segments updated.
        """
        try:
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
            # Find how many points were updated (optional, for return value)
            # For speed, we can skip counting or do a quick separate count query.
            # Let's just return 1 to indicate success for now or perform a count.
            return 1
        except Exception as e:
            log(f"set_speaker_name failed: {e}")
            return 0

    def set_speaker_main(
        self, cluster_id: int, segment_id: str, is_main: bool = True
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
                    collection_name=self.VOICE_COLLECTION,
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
                collection_name=self.VOICE_COLLECTION,
                payload={"is_main": is_main},
                points=models.PointIdsList(points=[segment_id]),
            )
            return True
        except Exception as e:
            log(f"set_speaker_main failed: {e}")
            return False

    # =========================================================================
    # SCENE-LEVEL STORAGE (Production architecture like Twelve Labs)
    # =========================================================================

    @observe("db_store_scene")
    async def store_scene(
        self,
        media_path: str,
        start_time: float,
        end_time: float,
        visual_text: str = "",
        motion_text: str = "",
        dialogue_text: str = "",
        visual_features: list[float]
        | None = None,  # Actual visual embedding (CLIP/SigLIP)
        internvideo_features: list[float]
        | None = None,  # InternVideo action embedding
        languagebind_features: list[float]
        | None = None,  # LanguageBind multimodal embedding
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Store a scene with multi-vector embeddings (visual, motion, dialogue).

        This is the production-grade approach used by Twelve Labs Marengo.
        Each scene gets 6 vectors:
        - visual: Text embedding of visual description
        - motion: Text embedding of motion/action description
        - dialogue: Text embedding of dialogue/transcript
        - visual_features: ACTUAL visual embedding from CLIP/SigLIP (for image-as-query)
        - internvideo: InternVideo2 action embedding (for action queries)
        - languagebind: LanguageBind multimodal embedding (text-aligned video)

        Args:
            media_path: Path to the source video.
            start_time: Scene start timestamp in seconds.
            end_time: Scene end timestamp in seconds.
            visual_text: Text describing visual content (entities, clothing, people).
            motion_text: Text describing actions and movement.
            dialogue_text: Transcript/dialogue for this scene.
            visual_features: Optional actual visual embedding from visual encoder.
            internvideo_features: Optional InternVideo action embedding.
            languagebind_features: Optional LanguageBind multimodal embedding.
            payload: Additional structured data (SceneData.to_payload()).

        Returns:
            The generated scene ID.
        """
        # Generate multi-vector embeddings (text-based)
        visual_vec = (await self.encode_texts(visual_text or "scene"))[0]
        motion_vec = (await self.encode_texts(motion_text or "activity"))[0]
        dialogue_vec = (await self.encode_texts(dialogue_text or "silence"))[0]

        # Normalize empty texts
        visual_vec = visual_vec if visual_text else [0.0] * self.TEXT_DIM
        motion_vec = motion_vec if motion_text else [0.0] * self.TEXT_DIM
        dialogue_vec = dialogue_vec if dialogue_text else [0.0] * self.TEXT_DIM

        # Visual features (actual visual embedding) - use placeholder if not provided
        visual_features_dim = getattr(settings, "visual_features_dim", 768)
        video_embedding_dim = getattr(settings, "video_embedding_dim", 1024)

        if visual_features is None:
            visual_features = [0.0] * visual_features_dim
        elif len(visual_features) != visual_features_dim:
            log(
                f"Visual features dim mismatch: got {len(visual_features)}, expected {visual_features_dim}",
                level="ERROR",
            )
            visual_features = [0.0] * visual_features_dim

        if internvideo_features is None:
            internvideo_features = [0.0] * video_embedding_dim
        elif len(internvideo_features) != video_embedding_dim:
            log(
                f"InternVideo features dim mismatch: got {len(internvideo_features)}, expected {video_embedding_dim}",
                level="ERROR",
            )
            internvideo_features = [0.0] * video_embedding_dim

        if languagebind_features is None:
            languagebind_features = [0.0] * video_embedding_dim
        elif len(languagebind_features) != video_embedding_dim:
            log(
                f"LanguageBind features dim mismatch: got {len(languagebind_features)}, expected {video_embedding_dim}",
                level="ERROR",
            )
            languagebind_features = [0.0] * video_embedding_dim

        # Generate unique scene ID
        scene_key = f"{media_path}_{start_time:.3f}_{end_time:.3f}"
        scene_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, scene_key))

        # Build full payload
        full_payload = {
            "media_path": media_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "visual_text": visual_text,
            "motion_text": motion_text,
            "dialogue_text": dialogue_text,
            "has_visual_features": visual_features is not None
            and any(v != 0 for v in visual_features[:10]),
            "has_internvideo": internvideo_features is not None
            and any(v != 0 for v in internvideo_features[:10]),
            "has_languagebind": languagebind_features is not None
            and any(v != 0 for v in languagebind_features[:10]),
        }
        if payload:
            full_payload.update(payload)

        # Prepare the vector dictionary
        vector_dict = {
            "visual": visual_vec,
            "motion": motion_vec,
            "dialogue": dialogue_vec,
            "internvideo": internvideo_features,
            "languagebind": languagebind_features,
        }

        # Conditionally add visual_features to the vector_dict
        if (
            visual_features is not None
            and len(visual_features) == visual_features_dim
        ):
            vector_dict["visual_features"] = visual_features
        else:
            # Fallback for missing or mismatched visual features (e.g. dependency missing)
            # Use zero vector of correct dim to prevent DB error
            zero_features = [0.0] * visual_features_dim
            vector_dict["visual_features"] = zero_features

        self.client.upsert(
            collection_name=self.SCENES_COLLECTION,
            points=[
                models.PointStruct(
                    id=scene_id,
                    vector=vector_dict,
                    payload=full_payload,
                )
            ],
        )

        log(
            f"Stored scene {start_time:.1f}-{end_time:.1f}s for {Path(media_path).name}"
        )
        return scene_id

    @observe("db_store_scenelet")
    async def store_scenelet(
        self,
        *,
        media_path: str,
        start_time: float,
        end_time: float,
        content_text: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Store a sliding window scenelet.

        Args:
            media_path: Source video path.
            start_time: Start timestamp.
            end_time: End timestamp.
            content_text: Fused text (Visual + Audio).
            payload: Additional metadata.
        """
        vector = (await self.encode_texts(content_text or "scenelet"))[0]

        scenelet_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{media_path}_sl_{start_time:.3f}")
        )

        full_payload = {
            "media_path": media_path,
            "start_time": start_time,
            "end_time": end_time,
            "text": content_text,
        }
        if payload:
            full_payload.update(payload)

        self.client.upsert(
            collection_name=self.SCENELETS_COLLECTION,
            points=[
                models.PointStruct(
                    id=scenelet_id,
                    vector={"content": vector},
                    payload=full_payload,
                )
            ],
        )
        return scenelet_id

    @observe("db_search_scenelets")
    @observe("db_search_voice_segments")
    async def search_voice_segments(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        video_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search voice segments using semantic vector similarity.

        Args:
            query: The search query.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.
            video_path: Optional filter by video path.

        Returns:
            List of matching voice segments with scores.
        """
        try:
            # Fix Async Error: Explicitly await and split
            # "TypeError: 'coroutine' object is not subscriptable"
            encoded_batches = await self.encode_texts(query, is_query=True)
            if not encoded_batches:
                return []
            query_vec = encoded_batches[0]

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
                collection_name=self.VOICE_COLLECTION,
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
                        "type": "voice_segment",
                        "speaker_id": payload.get("speaker_id"),
                        "speaker_name": payload.get(
                            "speaker_name", "Unknown Speaker"
                        ),
                        "text": payload.get(
                            "text", payload.get("transcription", "")
                        ),
                        "start": payload.get("start", 0),
                        "end": payload.get("end", 0),
                        "video_path": payload.get("media_path"),
                        **payload,
                    }
                )
            return results
        except Exception as e:
            log(f"search_voice_segments failed: {e}")
            return []

    @observe("db_search_audio_events")
    async def search_audio_events(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        video_path: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = []
        conditions.append(
            models.FieldCondition(
                key="event_class",
                match=models.MatchText(text=query),
            )
        )
        if video_path:
            conditions.append(
                models.FieldCondition(
                    key="media_path",
                    match=models.MatchValue(value=video_path),
                )
            )

        try:
            resp, _ = self.client.scroll(
                collection_name=self.AUDIO_EVENTS_COLLECTION,
                scroll_filter=models.Filter(must=conditions),
                limit=limit,
            )
            return [
                {"id": str(p.id), "score": 1.0, **(p.payload or {})}
                for p in resp
            ]
        except Exception as e:
            log(f"Audio event search failed: {e}")
            return []

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

    @observe("db_search_audio_events_semantic")
    async def search_audio_events_semantic(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        video_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search audio events using semantic vector similarity via CLAP.

        Uses CLAP text encoder to match against stored CLAP audio embeddings.
        Unlike search_audio_events which does text matching on event_class,
        this method performs vector-based semantic search for more flexible
        audio event discovery (e.g., "sudden loud noise" can match "explosion").

        Args:
            query: The search query.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.
            video_path: Optional filter by video path.

        Returns:
            List of matching audio events with scores.
        """
        try:
            # Use CLAP text encoder to get 512-dim embedding for audio search
            from core.processing.audio_events import get_audio_detector

            audio_detector = get_audio_detector()
            query_vec = await audio_detector.encode_text(query)

            if query_vec is None:
                log(
                    "CLAP text encoder unavailable, falling back to text search"
                )
                # Fallback to text-based search
                return await self.search_audio_events(
                    query=query,
                    limit=limit,
                    video_path=video_path,
                )

            conditions: list[models.Condition] = []
            # Only search events with actual embeddings
            conditions.append(
                models.FieldCondition(
                    key="has_embedding",
                    match=models.MatchValue(value=True),
                )
            )
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
                collection_name=self.AUDIO_EVENTS_COLLECTION,
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
                        "type": "audio_event",
                        "label": payload.get(
                            "event", payload.get("label", "audio")
                        ),
                        "start": payload.get("start_time", 0),
                        "end": payload.get("end_time", 0),
                        "video_path": payload.get("media_path"),
                        "confidence": payload.get("confidence", 0),
                        **payload,
                    }
                )
            return results
        except Exception as e:
            log(f"search_audio_events_semantic failed: {e}")
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
            query_vec = (await self.encode_texts(query, is_query=True))[0]

            resp = self.client.query_points(
                collection_name=self.VIDEO_METADATA_COLLECTION,
                query=query_vec,
                limit=limit,
                score_threshold=score_threshold,
            )

            results = []
            for hit in resp.points:
                payload = hit.payload or {}
                results.append(
                    {
                        "id": str(hit.id),
                        "score": hit.score,
                        "type": "video_metadata",
                        "video_path": payload.get("video_path"),
                        "summary": payload.get("summary"),
                        "title": payload.get("title"),
                        **payload,
                    }
                )
            return results
        except Exception as e:
            log(f"search_video_metadata failed: {e}")
            return []

    @observe("db_search_scenes")
    async def search_scenes(
        self,
        query: str | list[float],
        *,
        limit: int = 20,
        score_threshold: float | None = None,
        # Identity filters - FIX: Accept list of names instead of single name
        person_names: list[str] | None = None,
        person_name: str | None = None,  # DEPRECATED: kept for backwards compat
        face_cluster_ids: list[int] | None = None,
        # Clothing/appearance filters (for complex queries)
        clothing_color: str | None = None,
        clothing_type: str | None = None,
        accessories: list[str] | None = None,
        # Content filters
        location: str | None = None,
        visible_text: list[str] | None = None,
        # Action filters
        action_keywords: list[str] | None = None,
        # Deep Research Filters
        mood: str | None = None,
        shot_type: str | None = None,
        aesthetic_score: float | None = None,
        # Time filters
        video_path: str | None = None,
        video_paths: list[str] | None = None,
        # Search mode
        search_mode: str = "hybrid",  # "visual", "motion", "dialogue", "hybrid"
    ) -> list[dict[str, Any]]:
        """Search scenes with comprehensive filtering for complex queries.

        Supports queries like:
        - "Prakash wearing blue shirt bowling at Brunswick hitting a strike"

        Args:
            query: Natural language search query.
            limit: Maximum results.
            score_threshold: Minimum similarity score.
            person_names: Filter by person names (list). Matches ANY name in list.
            person_name: DEPRECATED. Use person_names instead.
            face_cluster_ids: Filter by face clusters.
            clothing_color: Filter by clothing color (e.g., "blue").
            clothing_type: Filter by clothing type (e.g., "shirt").
            accessories: Filter by accessories (e.g., ["spectacles"]).
            location: Filter by location (e.g., "Brunswick").
            visible_text: Filter by visible text/brands.
            action_keywords: Filter by actions.
            video_path: Filter by specific video.
            search_mode: Which vector(s) to search.

        Returns:
            List of matching scenes with timestamps and metadata.
        """
        # Build query vector based on mode
        if isinstance(query, str):
            query_vec = (await self.encode_texts(query, is_query=True))[0]
        else:
            query_vec = query

        # Build filter conditions
        conditions: list[models.Condition] = []

        # Video filter
        if video_path:
            conditions.append(
                models.FieldCondition(
                    key="media_path",
                    match=models.MatchValue(value=video_path),
                )
            )

        # Hierarchical Filter (Multiple Videos)
        if video_paths:
            conditions.append(
                models.FieldCondition(
                    key="media_path",
                    match=models.MatchAny(any=video_paths),
                )
            )

        # Identity filter - FIX: Support multiple person names
        # Merge person_names list with deprecated person_name for backwards compat
        all_person_names = list(person_names) if person_names else []
        if person_name and person_name not in all_person_names:
            all_person_names.append(person_name)

        if all_person_names:
            conditions.append(
                models.FieldCondition(
                    key="person_names",
                    match=models.MatchAny(
                        any=all_person_names
                    ),  # Match ANY name
                )
            )

        if face_cluster_ids:
            conditions.append(
                models.FieldCondition(
                    key="face_cluster_ids",
                    match=models.MatchAny(any=face_cluster_ids),
                )
            )

        # Clothing/appearance filters (critical for complex queries)
        if clothing_color:
            conditions.append(
                models.FieldCondition(
                    key="clothing_colors",
                    match=models.MatchAny(any=[clothing_color.lower()]),
                )
            )

        if clothing_type:
            conditions.append(
                models.FieldCondition(
                    key="clothing_types",
                    match=models.MatchAny(any=[clothing_type.lower()]),
                )
            )

        if accessories:
            conditions.append(
                models.FieldCondition(
                    key="accessories",
                    match=models.MatchAny(any=accessories),
                )
            )

        # Location filter
        if location:
            conditions.append(
                models.FieldCondition(
                    key="location",
                    match=models.MatchText(text=location),
                )
            )

        # Visible text/brand filter
        if visible_text:
            conditions.append(
                models.FieldCondition(
                    key="visible_text",
                    match=models.MatchAny(any=visible_text),
                )
            )

        # Action filter
        if action_keywords:
            conditions.append(
                models.FieldCondition(
                    key="actions",
                    match=models.MatchAny(any=action_keywords),
                )
            )

        # Deep Research Filters (Cinematography)
        if mood:
            conditions.append(
                models.FieldCondition(
                    key="mood", match=models.MatchAny(any=[mood])
                )
            )

        if shot_type:
            conditions.append(
                models.FieldCondition(
                    key="shot_type", match=models.MatchAny(any=[shot_type])
                )
            )

        if aesthetic_score is not None:
            conditions.append(
                models.FieldCondition(
                    key="aesthetic_score",
                    range=models.Range(gte=aesthetic_score),
                )
            )

        # Build final filter
        query_filter = models.Filter(must=conditions) if conditions else None

        # Execute search based on mode
        results = []

        if search_mode == "hybrid":
            # Search all enabled vectors and combine results
            # Core text vectors (always searched)
            target_vectors = ["visual", "motion", "dialogue"]

            # Video understanding vectors (if enabled and stored)
            if getattr(settings, "enable_video_embeddings", True):
                target_vectors.extend(["internvideo", "languagebind"])

            # Visual features (CLIP/SigLIP) if enabled
            if getattr(settings, "enable_visual_features", True):
                target_vectors.append("visual_features")

            for vector_name in target_vectors:
                try:
                    # For video/visual vectors, add filter to only search scenes with those embeddings
                    search_filter = query_filter
                    if vector_name in [
                        "internvideo",
                        "languagebind",
                        "visual_features",
                    ]:
                        has_key = f"has_{vector_name}"
                        embedding_filter = models.FieldCondition(
                            key=has_key,
                            match=models.MatchValue(value=True),
                        )
                        if search_filter:
                            # Combine with existing filter
                            search_filter = models.Filter(
                                must=[*search_filter.must, embedding_filter]
                                if search_filter.must
                                else [embedding_filter]
                            )
                        else:
                            search_filter = models.Filter(
                                must=[embedding_filter]
                            )

                    resp = self.client.query_points(
                        collection_name=self.SCENES_COLLECTION,
                        query=query_vec,
                        using=vector_name,
                        limit=limit,
                        score_threshold=score_threshold,
                        query_filter=search_filter,
                    )
                    for hit in resp.points:
                        results.append(
                            {
                                "score": hit.score,
                                "id": str(hit.id),
                                "vector_type": vector_name,
                                **(hit.payload or {}),
                            }
                        )
                except Exception as e:
                    log(f"Scene search ({vector_name}) error: {e}")

            # Dedupe and sort by highest score
            seen_ids = set()
            unique_results = []
            for r in sorted(results, key=lambda x: x["score"], reverse=True):
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    unique_results.append(r)
            results = unique_results[:limit]

        else:
            # Single vector search
            try:
                resp = self.client.query_points(
                    collection_name=self.SCENES_COLLECTION,
                    query=query_vec,
                    using=search_mode,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
                for hit in resp.points:
                    results.append(
                        {
                            "score": hit.score,
                            "id": str(hit.id),
                            "vector_type": search_mode,
                            **(hit.payload or {}),
                        }
                    )
            except Exception as e:
                log(f"Scene search ({search_mode}) error: {e}")

        # HITL Name Confidence Boosting
        # Boost scores for results containing HITL-named identities that match the query
        # FIX: Use all_person_names list instead of single person_name
        if all_person_names:
            query_names_lower = [n.lower() for n in all_person_names]
            for result in results:
                # Check if result has face_names or person_names that match
                face_names = result.get("face_names", []) or result.get(
                    "person_names", []
                )
                if face_names:
                    for name in face_names:
                        if name and any(
                            qn in name.lower() for qn in query_names_lower
                        ):
                            # 50% boost for exact HITL name match
                            result["score"] = result.get("score", 0) * 1.5
                            result["hitl_boost"] = True
                            result["matched_person"] = (
                                name  # Track which person matched
                            )
                            break

        # Re-sort after boosting
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return results

    @observe("db_search_scenes_by_image")
    async def search_scenes_by_image(
        self,
        image: np.ndarray | bytes | Path,
        limit: int = 10,
        video_path: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Search scenes using an image as query (true visual search).

        This enables queries like "find scenes that look like this image"
        by using actual visual embeddings (CLIP/SigLIP) instead of text.

        Args:
            image: Query image (numpy array, bytes, or path).
            limit: Maximum results.
            video_path: Optional filter by video.
            score_threshold: Minimum similarity.

        Returns:
            List of matching scenes with scores.
        """
        from core.processing.visual_encoder import get_default_visual_encoder

        # Encode query image
        encoder = get_default_visual_encoder()
        query_vector = await encoder.encode_image(image)

        # Build filter
        filters = []
        if video_path:
            filters.append(
                models.FieldCondition(
                    key="media_path",
                    match=models.MatchValue(value=video_path),
                )
            )
        # Only search scenes with actual visual features
        filters.append(
            models.FieldCondition(
                key="has_visual_features",
                match=models.MatchValue(value=True),
            )
        )

        try:
            results = self.client.search(
                collection_name=self.SCENES_COLLECTION,
                query_vector=models.NamedVector(
                    name="visual_features",
                    vector=query_vector,
                ),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=models.Filter(must=filters) if filters else None,
            )

            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "media_path": r.payload.get("media_path", ""),
                    "start_time": r.payload.get("start_time", 0),
                    "end_time": r.payload.get("end_time", 0),
                    "visual_text": r.payload.get("visual_text", ""),
                    "search_mode": "image",
                    **r.payload,
                }
                for r in results
                if r.payload
            ]
        except Exception as e:
            log(f"Image search failed: {e}")
            return []

    @observe("db_search_scenes_by_action")
    async def search_scenes_by_action(
        self,
        query: str,
        limit: int = 10,
        video_path: str | None = None,
        score_threshold: float | None = None,
        search_mode: str = "hybrid",  # "internvideo", "languagebind", "hybrid"
    ) -> list[dict]:
        """Search scenes by action/motion using video embeddings.

        This uses InternVideo (action recognition) and LanguageBind (multimodal)
        embeddings for queries like "person kicking ball" or "car driving fast".

        Args:
            query: Action/motion query text.
            limit: Maximum results.
            video_path: Optional filter by video.
            score_threshold: Minimum similarity.
            search_mode: Which embedding to use (internvideo/languagebind/hybrid).

        Returns:
            List of matching scenes with scores.
        """
        from core.processing.video_understanding import LanguageBindEncoder

        # Build filter
        filters = []
        if video_path:
            filters.append(
                models.FieldCondition(
                    key="media_path",
                    match=models.MatchValue(value=video_path),
                )
            )

        results = []

        # Get text embedding for query (LanguageBind is text-aligned)
        try:
            encoder = LanguageBindEncoder()
            query_embedding = await encoder.encode_text(query)

            if query_embedding is None:
                # Fallback to standard text encoding
                query_embedding = (await self.encode_texts(query))[0]

            query_embedding = (
                list(query_embedding) if query_embedding is not None else None
            )
        except Exception as e:
            log(f"Failed to encode action query: {e}")
            query_embedding = (await self.encode_texts(query))[0]

        if query_embedding is None:
            return []

        # Search based on mode
        if search_mode in ("languagebind", "hybrid"):
            try:
                # Add filter for scenes with LanguageBind embeddings
                lb_filters = filters.copy()
                lb_filters.append(
                    models.FieldCondition(
                        key="has_languagebind",
                        match=models.MatchValue(value=True),
                    )
                )

                lb_results = self.client.search(
                    collection_name=self.SCENES_COLLECTION,
                    query_vector=models.NamedVector(
                        name="languagebind",
                        vector=query_embedding,
                    ),
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=models.Filter(must=lb_filters)
                    if lb_filters
                    else None,
                )

                for r in lb_results:
                    if r.payload:
                        results.append(
                            {
                                "id": str(r.id),
                                "score": r.score,
                                "media_path": r.payload.get("media_path", ""),
                                "start_time": r.payload.get("start_time", 0),
                                "end_time": r.payload.get("end_time", 0),
                                "motion_text": r.payload.get("motion_text", ""),
                                "search_mode": "languagebind",
                                "source": "languagebind",
                                **r.payload,
                            }
                        )
            except Exception as e:
                log(f"LanguageBind search failed: {e}")

        if search_mode in ("internvideo", "hybrid"):
            try:
                # Add filter for scenes with InternVideo embeddings
                iv_filters = filters.copy()
                iv_filters.append(
                    models.FieldCondition(
                        key="has_internvideo",
                        match=models.MatchValue(value=True),
                    )
                )

                iv_results = self.client.search(
                    collection_name=self.SCENES_COLLECTION,
                    query_vector=models.NamedVector(
                        name="internvideo",
                        vector=query_embedding,
                    ),
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=models.Filter(must=iv_filters)
                    if iv_filters
                    else None,
                )

                for r in iv_results:
                    if r.payload:
                        results.append(
                            {
                                "id": str(r.id),
                                "score": r.score,
                                "media_path": r.payload.get("media_path", ""),
                                "start_time": r.payload.get("start_time", 0),
                                "end_time": r.payload.get("end_time", 0),
                                "motion_text": r.payload.get("motion_text", ""),
                                "search_mode": "internvideo",
                                "source": "internvideo",
                                **r.payload,
                            }
                        )
            except Exception as e:
                log(f"InternVideo search failed: {e}")

        # RRF fusion if hybrid
        if search_mode == "hybrid" and results:
            # Deduplicate by ID, keeping highest score
            seen = {}
            for r in results:
                rid = r["id"]
                if rid not in seen or r["score"] > seen[rid]["score"]:
                    seen[rid] = r
            results = sorted(
                seen.values(), key=lambda x: x["score"], reverse=True
            )

        return results[:limit]

    @observe("db_explainable_search")
    def explainable_search(
        self,
        query_text: str,
        parsed_query: Any = None,
        limit: int = 10,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search with explainable results - returns reasoning for each match.

        This is the SOTA search method that provides:
        - Matched entities with individual confidence scores
        - Reasoning for why the result was selected
        - Face/voice identification with names
        - Timestamp accuracy justification

        Args:
            query_text: The search query text.
            parsed_query: Optional DynamicParsedQuery with extracted entities.
            limit: Maximum results to return.
            score_threshold: Minimum similarity score.

        Returns:
            List of results with explainable metadata:
            [
                {
                    "id": "...",
                    "score": 0.85,
                    "timestamp": 45.2,
                    "segment_url": "/media/segment?...",
                    "matched_entities": {
                        "person": {"name": "Prakash", "confidence": 0.98, "source": "face"},
                        "clothing": [{"item": "blue t-shirt", "confidence": 0.87}],
                        "action": {"name": "bowling", "confidence": 0.92}
                    },
                    "reasoning": "Frame shows Prakash (face ID #3) in blue upper garment...",
                    "evidence": ["face_match", "color_match", "action_match"]
                }
            ]
        """
        # 1. Generate embedding for query
        self.get_embedding(query_text)

        # 2. Perform multi-vector search
        raw_results = self.search_scenes(
            query=query_text,
            limit=limit * 2,  # Get more candidates for re-ranking
            score_threshold=score_threshold,
            search_mode="hybrid",
        )

        # 3. Enrich each result with explainable metadata
        explainable_results = []

        for result in raw_results[:limit]:
            # Extract entities from the result payload
            matched_entities = {}
            evidence = []
            reasoning_parts = []

            # Check for person/face matches
            face_names = result.get("face_names", []) or result.get(
                "person_names", []
            )
            face_ids = result.get("face_ids", [])
            if face_names:
                for i, name in enumerate(face_names):
                    if name:
                        matched_entities["person"] = {
                            "name": name,
                            "confidence": 0.95,  # Face recognition typically high confidence
                            "source": "face_recognition",
                            "face_id": face_ids[i]
                            if i < len(face_ids)
                            else None,
                        }
                        evidence.append("face_match")
                        reasoning_parts.append(
                            f"Identified {name} via face recognition"
                        )
                        break

            # Check for voice matches
            voice_names = result.get("voice_names", []) or result.get(
                "speaker_names", []
            )
            voice_ids = result.get("voice_ids", [])
            if voice_names:
                for i, name in enumerate(voice_names):
                    if name:
                        matched_entities["voice"] = {
                            "name": name,
                            "confidence": 0.85,
                            "source": "voice_diarization",
                            "voice_id": voice_ids[i]
                            if i < len(voice_ids)
                            else None,
                        }
                        evidence.append("voice_match")
                        reasoning_parts.append(f"Voice identified as {name}")
                        break

            # Check for text/OCR matches
            visible_text = result.get("visible_text", []) or result.get(
                "ocr_text", []
            )
            if visible_text:
                matched_entities["text"] = {
                    "items": visible_text[:5],  # Top 5 text items
                    "confidence": 0.90,
                    "source": "ocr",
                }
                evidence.append("text_match")
                reasoning_parts.append(
                    f"Visible text: {', '.join(visible_text[:3])}"
                )

            # Check for location
            location = result.get("location", "") or result.get(
                "scene_location", ""
            )
            if location:
                matched_entities["location"] = {
                    "name": location,
                    "confidence": 0.80,
                    "source": "scene_analysis",
                }
                evidence.append("location_match")
                reasoning_parts.append(f"Location: {location}")

            # Check for actions
            actions = result.get("actions", []) or result.get(
                "action_keywords", []
            )
            if actions:
                matched_entities["actions"] = {
                    "items": actions[:5],
                    "confidence": 0.75,
                    "source": "visual_analysis",
                }
                evidence.append("action_match")
                reasoning_parts.append(f"Actions: {', '.join(actions[:3])}")

            # Get description for additional context
            description = result.get("description", "") or result.get(
                "dense_caption", ""
            )

            # Build reasoning string
            reasoning = (
                "; ".join(reasoning_parts)
                if reasoning_parts
                else description[:200]
            )

            # Build explainable result
            explainable_results.append(
                {
                    "id": result.get("id"),
                    "score": result.get("score", 0),
                    "timestamp": result.get("start_time")
                    or result.get("timestamp", 0),
                    "end_time": result.get("end_time"),
                    "media_path": result.get("media_path"),
                    "matched_entities": matched_entities,
                    "reasoning": reasoning,
                    "evidence": evidence,
                    "hitl_boost": result.get("hitl_boost", False),
                    # Include raw data for debugging
                    "raw_description": description[:500]
                    if description
                    else None,
                }
            )

        return explainable_results

    def get_scene_by_id(self, scene_id: str) -> dict[str, Any] | None:
        """Get a scene by its ID.

        Args:
            scene_id: The scene ID.

        Returns:
            Scene data or None if not found.
        """
        try:
            points = self.client.retrieve(
                collection_name=self.SCENES_COLLECTION,
                ids=[scene_id],
                with_payload=True,
            )
            if points:
                return {"id": scene_id, **(points[0].payload or {})}
            return None
        except Exception:
            return None

    def get_scenes_for_video(
        self,
        video_path: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all scenes for a video, ordered by start time.

        Args:
            video_path: Path to the video.
            limit: Maximum results.

        Returns:
            List of scenes ordered by start_time.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.SCENES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="media_path",
                            match=models.MatchValue(value=video_path),
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
            )
            scenes = []
            for point in resp[0]:
                scenes.append(
                    {
                        "id": str(point.id),
                        **(point.payload or {}),
                    }
                )
            # Sort by start_time
            scenes.sort(key=lambda x: x.get("start_time", 0))
            return scenes
        except Exception:
            return []

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

            # Propagate to media frames
            self._propagate_face_name_to_frames(cluster_id, name)

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
                points=ids,  # type: ignore
            )

            # Propagate cluster ID change to media frames
            self._update_frames_cluster_rename(from_cluster, to_cluster)

            return len(ids)
        except Exception:
            return 0

    @observe("db_update_video_metadata")
    def update_video_metadata(
        self, video_path: str, metadata: dict[str, Any]
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

    def set_face_main(self, cluster_id: int, is_main: bool = True) -> bool:
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

            if not face_ids:
                log(f"[HITL] No faces found in cluster {cluster_id}")
                return False

            # Update all faces in cluster with PointIdsList (correct API usage)
            self.client.set_payload(
                collection_name=self.FACES_COLLECTION,
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

    def get_faces_by_media(
        self, media_path: str, limit: int = 1000
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
                results.append(
                    {
                        "id": point.id,
                        "media_path": payload.get("media_path"),
                        "start": payload.get("start"),
                        "end": payload.get("end"),
                        "speaker_label": payload.get("speaker_label"),
                        "audio_path": payload.get("audio_path"),
                    }
                )
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
                    "vectors_count": getattr(
                        info, "vectors_count", info.points_count
                    ),
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

    @observe("db_get_faces_grouped_by_cluster")
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

    @observe("db_get_all_cluster_centroids")
    def get_all_cluster_centroids(self) -> dict[int, list[float]]:
        """Get cluster centroids for global identity matching.

        Returns ONE embedding per cluster_id (the mean of all faces in that cluster).
        This is O(1) per match instead of O(N) when matching new faces.

        Only returns clusters with at least one named face (HITL verified).
        """
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
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

    @observe("db_update_cluster_centroid")
    def update_cluster_centroid(
        self, cluster_id: int, new_embedding: list[float], alpha: float = 0.3
    ) -> bool:
        """Update cluster centroid with exponential moving average.

        Args:
            cluster_id: Cluster to update.
            new_embedding: New face embedding to incorporate.
            alpha: EMA weight for new embedding (0.3 = 30% new, 70% old).
        """
        return True

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

    @observe("db_delete_voice_cluster")
    def delete_voice_cluster(self, cluster_id: int) -> int:
        """Delete an entire voice cluster and all its segments.

        Args:
            cluster_id: The voice cluster ID to delete.

        Returns:
            Number of segments deleted.
        """
        try:
            # 1. Get all segments in this cluster
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
                collection_name=self.VOICE_COLLECTION,
                points_selector=models.PointIdsList(points=point_ids),
            )
            log(
                f"[DB] Deleted voice cluster {cluster_id}: {len(point_ids)} segments"
            )
            return len(point_ids)
        except Exception as e:
            log(f"delete_voice_cluster failed: {e}")
            return 0

    @observe("db_delete_face_cluster")
    def delete_face_cluster(self, cluster_id: int) -> int:
        """Delete an entire face cluster and all its faces.

        Args:
            cluster_id: The face cluster ID to delete.

        Returns:
            Number of faces deleted.
        """
        try:
            # 1. Get all faces in this cluster
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
                collection_name=self.FACES_COLLECTION,
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
    def get_all_voice_segments(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Retrieve all voice segments for listing/management.

        Args:
            limit: Maximum number of segments to return.

        Returns:
            List of voice segments.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.VOICE_COLLECTION,
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

    def get_face_name_by_cluster(self, cluster_id: str | int) -> str | None:
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
                with_payload=["cluster_id"],
            )
            if resp[0]:
                return (resp[0][0].payload or {}).get("cluster_id")
            return None
        except Exception:
            return None

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
                collection_name=self.VOICE_COLLECTION,
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
                collection_name=self.FACES_COLLECTION,
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
                collection_name=self.FACES_COLLECTION,
                payload={"name": name},
                points=point_ids,  # type: ignore
            )

            # === Step 4: Propagate name to frames for proper search ===
            self._propagate_face_name_to_frames(cluster_id, name)

            # === Step 5: Identity Linking (Face + Voice Cross-Modal) ===
            try:
                from core.storage.identity_graph import identity_graph

                # Get/Create Global Identity
                identity = identity_graph.get_or_create_identity_by_name(name)

                # Link these face tracks to the identity
                identity_graph.link_faces_to_identity(point_ids, identity.id)
                log(
                    f"[Identity] Linked {len(point_ids)} faces to {identity.name} ({identity.id})"
                )

                # === Cross-Modal Link: Check for voice cluster with same name ===
                voice_cluster = self.get_speaker_cluster_by_name(name)
                if voice_cluster:
                    log(
                        f"[Identity] Found voice cluster with same name '{name}' (ID: {voice_cluster})"
                    )
                    # Get voice segment IDs for this cluster
                    voice_resp = self.client.scroll(
                        collection_name=self.VOICE_COLLECTION,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="voice_cluster_id",
                                    match=models.MatchValue(
                                        value=voice_cluster
                                    ),
                                )
                            ]
                        ),
                        limit=1000,
                    )
                    voice_ids = [str(p.id) for p in voice_resp[0]]
                    if voice_ids:
                        identity_graph.link_voices_to_identity(
                            voice_ids, identity.id
                        )
                        log(
                            f"[Identity] Cross-linked {len(voice_ids)} voice segments to {identity.name}"
                        )

            except Exception as e:
                log(f"[Identity] Linking failed: {e}")
            # ------------------------

            log(
                f"[HITL] Set name '{name}' on {len(point_ids)} faces in cluster {cluster_id}"
            )
            return len(point_ids)
        except Exception as e:
            log(f"set_face_name failed: {e}")
            return 0

    def re_embed_face_cluster_frames(
        self, cluster_id: int, new_name: str
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

    def _propagate_speaker_name_to_frames(
        self, cluster_id: int, name: str
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
                collection_name=self.VOICE_COLLECTION,
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
                    collection_name=self.MEDIA_COLLECTION,
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

    def get_frames_by_face_cluster(
        self, cluster_id: int, limit: int = 1000
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

    @observe("db_search_voice")
    @observe("db_search_hybrid_legacy")
    async def search_frames_hybrid_legacy(
        self,
        query: str,
        video_paths: str | list[str] | None = None,
        limit: int = 20,
        weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Legacy hybrid search with keyword boosting.

        NOTE: Main search_frames_hybrid is at line ~826 using RRF algorithm.
        This version uses simpler keyword boosting approach.

        Args:
            query: Natural language search query.
            video_paths: Optional filter to specific video(s).
            limit: Maximum results to return.
            weights: Optional dictionary of boosting weights.

        Returns:
            Ranked list of matching frames with scores.
        """
        # Default Weights (Tuned for balanced precision/recall)
        w = {
            "face_match": 0.20,
            "speaker_match": 0.15,
            "entity_match": 0.10,
            "text_match": 0.08,
            "scene_match": 0.08,
            "action_match": 0.05,
        }
        if weights:
            w.update(weights)

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
                log(
                    f"[HybridSearch] Identity filter: {matched_names} → clusters {cluster_ids}"
                )

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
        query_vector = (await self.encode_texts(query, is_query=True))[0]

        vector_results = self.client.query_points(
            collection_name=self.MEDIA_COLLECTION,
            query=query_vector,
            limit=limit * 3,
            query_filter=combined_filter,
        )

        # 4. Extract keywords for boosting
        query_words = {w.lower() for w in query.split() if len(w) > 2}
        # Remove common words
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "are",
            "was",
            "were",
        }
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
                    boost += w["face_match"]

            # Check speaker_names
            for name in payload.get("speaker_names", []):
                if name and name.lower() in query_lower:
                    boost += w["speaker_match"]

            # Check entities
            for entity in payload.get("entities", []):
                if entity and any(w in entity.lower() for w in query_words):
                    boost += w["entity_match"]

            # Check visible_text
            for text in payload.get("visible_text", []):
                if text and any(w in text.lower() for w in query_words):
                    boost += w["text_match"]

            # Check scene_location
            location = payload.get("scene_location", "") or ""
            if any(w in location.lower() for w in query_words):
                boost += w["scene_match"]

            # Check action/description
            action = (
                payload.get("action", "")
                or payload.get("description", "")
                or ""
            )
            if any(w in action.lower() for w in query_words):
                boost += w["action_match"]

            results.append(
                {
                    "id": str(hit.id),
                    "score": score + boost,
                    "base_score": score,
                    "keyword_boost": boost,
                    **payload,
                }
            )

        # 6. Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)

        log(
            f"[HybridSearch] Query: '{query}' | Identity filter: {bool(identity_filter)} | Results: {len(results)}"
        )

        return results[:limit]

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

    def re_embed_voice_cluster_frames(
        self, cluster_id: int, new_name: str, old_name: str | None = None
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
                collection_name=self.VOICE_COLLECTION,
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
                    collection_name=self.MEDIA_COLLECTION,
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

    @observe("db_delete_media")
    def delete_media_by_path(self, media_path: str) -> None:
        """Delete all data associated with a media file."""
        for collection in [
            self.MEDIA_SEGMENTS_COLLECTION,
            self.MEDIA_COLLECTION,
            self.FACES_COLLECTION,
            self.VOICE_COLLECTION,
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

    def store_scene_metadata(self, media_path: str, scenes: list[dict]) -> None:
        """Stores scene-level metadata for a video.

        Note: Current implementation only logs the receipt of scenes.

        Args:
            media_path: Path to the source video.
            scenes: List of scene data dictionaries.
        """
        log(
            f"Received {len(scenes)} scenes for {media_path} (Storage not implemented)"
        )

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
                collection_name=self.FACES_COLLECTION,
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
                collection_name=self.FACES_COLLECTION,
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
                collection_name=self.FACES_COLLECTION,
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
                collection_name=self.MEDIA_COLLECTION,
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
                    collection_name=self.MEDIA_COLLECTION,
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
                collection_name=self.FACES_COLLECTION,
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
                    collection_name=self.FACES_COLLECTION,
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
                collection_name=self.MEDIA_COLLECTION,
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
                    collection_name=self.MEDIA_COLLECTION,
                    payload={"face_names": names},
                    points=[str(p.id)],
                )
                updated += 1
            return updated
        except Exception:
            return 0

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

    def get_unresolved_voices(self, limit: int = 100) -> list[dict]:
        """Get voice segments that are part of unnamed clusters.

        Returns:
            List of voice segment dictionaries with flat structure.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.VOICE_COLLECTION,
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

    def _merge_face_cluster_frames(self, source_id: int, target_id: int):
        """Helper to update frame references when merging face clusters."""
        try:
            # Scroll frames that have source_id
            resp = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
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
                        collection_name=self.MEDIA_COLLECTION,
                        payload={"face_cluster_ids": new_ids},
                        points=[p.id],
                    )
        except Exception as e:
            log(f"_merge_face_cluster_frames failed: {e}")
