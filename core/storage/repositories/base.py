"""Base repository class with shared Qdrant client access."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.storage.constants import (
    FACE_VECTOR_SIZE,
    MEDIA_VECTOR_SIZE,
    TEXT_DIM,
    VOICE_VECTOR_SIZE,
)
from core.storage.qdrant_utils import paginated_scroll, sanitize_numpy_types
from core.utils.logger import log


class BaseRepository:
    """Base class for all Qdrant-based repositories.

    Provides shared access to the Qdrant client, dim validation,
    and common query helpers.
    """

    def __init__(self, client: QdrantClient) -> None:
        self.client = client
        self._expected_dims: dict[str, int] = {}

    def _validate_vector_dim(
        self, vector: list | None, collection: str, context: str = ""
    ) -> bool:
        """Validate vector dimension before insert."""
        if vector is None:
            return True
        expected = self._expected_dims.get(collection)
        if expected is None:
            return True
        actual = len(vector)
        if actual != expected:
            log(
                f"[DIM MISMATCH] {collection}: expected {expected}d, got {actual}d. {context}",
                level="ERROR",
            )
            return False
        return True

    def _scroll_all(
        self,
        collection_name: str,
        scroll_filter: models.Filter | None = None,
        limit: int = 1000,
        with_payload: bool = True,
        with_vectors: bool = False,
        payload_fields: list[str] | None = None,
    ) -> list:
        """Scroll through all matching points in a collection."""
        results = []
        offset = None
        remaining = limit

        while remaining > 0:
            batch_size = min(500, remaining)
            batch, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=batch_size,
                offset=offset,
                with_payload=payload_fields if payload_fields else with_payload,
                with_vectors=with_vectors,
            )
            results.extend(batch)
            remaining -= len(batch)
            if offset is None or len(batch) < batch_size:
                break

        return results

    @staticmethod
    def _new_uuid() -> str:
        return str(uuid.uuid4())
