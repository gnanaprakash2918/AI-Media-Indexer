"""Shared utilities for Qdrant storage operations."""

from __future__ import annotations

import time
from functools import wraps
from typing import Any

from qdrant_client.http import models

from core.utils.logger import log


def retry_on_connection_error(max_retries: int = 3, delay: float = 1.0):
    """Retry Qdrant operations on connection errors."""

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

    Prevents serialization errors when upserting to Qdrant.
    Also detects unawaited coroutines.
    """
    import asyncio
    import inspect

    import numpy as np

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
    if isinstance(obj, list):
        return [sanitize_numpy_types(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
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
    """Paginated scroll for large result sets.

    Fetches results in batches using cursor-based pagination
    to prevent memory issues.
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
