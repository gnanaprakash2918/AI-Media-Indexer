"""Qdrant client handler and connection management."""

import time
from functools import wraps
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import settings
from core.utils.logger import log


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
                        time.sleep(delay * (attempt + 1))
            if last_error:
                raise last_error
            return None  # Should not happen

        return wrapper

    return decorator


def sanitize_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    import asyncio
    import inspect

    import numpy as np

    if asyncio.iscoroutine(obj) or inspect.iscoroutine(obj):
        log(f"[SANITIZE] ERROR: Unawaited coroutine: {obj}", level="ERROR")
        return "[ERROR: Unawaited coroutine]"

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


def paginated_scroll(
    client,
    collection_name: str,
    scroll_filter: models.Filter | None = None,
    limit: int = 1000,
    batch_size: int = 100,
    with_payload: bool = True,
    with_vectors: bool = False,
) -> list:
    """Generator-based paginated scroll for large result sets."""
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


class QdrantHandler:
    """Basic Qdrant connection and collection management."""

    def __init__(
        self,
        backend: str = settings.qdrant_backend,
        host: str = settings.qdrant_host,
        port: int = settings.qdrant_port,
        path: str = "qdrant_data_embedded",
    ):
        if backend == "memory":
            self.client = QdrantClient(path=path)
            log("Initialized embedded Qdrant", path=path, backend=backend)
        elif backend == "docker":
            # Simple, standard connection logic without giant loops.
            # Fix for Windows: Prefer 127.0.0.1 if host is localhost to avoid IPv6 issues.
            target_host = host
            if host == "localhost":
                target_host = "127.0.0.1"

            try:
                self.client = QdrantClient(host=target_host, port=port)
                # Verify connection immediately
                self.client.get_collections()
                log(
                    f"Connected to Qdrant at {target_host}:{port}",
                    backend=backend,
                )
            except Exception as exc:
                log(
                    f"Could not connect to Qdrant at {target_host}:{port}.",
                    error=str(exc),
                    host=host,
                    port=port,
                )
                raise ConnectionError("Qdrant connection failed.") from exc
        else:
            raise ValueError(
                f"Unknown backend: {backend!r} (use 'memory' or 'docker')"
            )

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

                if is_multi_vector and isinstance(vectors_config, dict):
                    pass
                elif not is_multi_vector and isinstance(vectors_config, dict):
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

    def create_index(
        self, collection_name: str, field_name: str, schema_type: Any
    ):
        """Wrapper for creating payload indexes."""
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )

    def _create_text_index(
        self, collection_name: str, field_name: str, **kwargs
    ):
        """Wrapper for text indexes."""
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.TextIndexParams(**kwargs),
        )
