"""Complex search, co-occurrence, and cross-domain queries.

Extracted from IdentityGraphManager to reduce God class size.
"""

from __future__ import annotations

import sqlite3
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from core.storage.identity_models import (
    FaceTrack,
)

if TYPE_CHECKING:
    pass


class IdentityQueryManager:
    """Complex search, co-occurrence, and cross-domain queries."""

    # These will be available via IdentityGraphManager inheritance
    db_path: str
    _lock: Lock

    # Type stub for type checkers
    def __getattr__(self, name: str) -> Any: ...

    def get_media_ids_for_identity(self, identity_id: str) -> list[str]:
        """Get all media IDs where an identity appears (for pre-filtering search)."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT media_id FROM (
                    SELECT media_id FROM face_tracks WHERE identity_id = ?
                    UNION
                    SELECT media_id FROM voice_tracks WHERE identity_id = ?
                )
            """,
                (identity_id, identity_id),
            )
            return [row[0] for row in cursor.fetchall()]

    def find_similar_face_tracks(
        self,
        embedding: list[float],
        threshold: float = 0.7,
        limit: int = 10,
    ) -> list[tuple[FaceTrack, float]]:
        """Find face tracks similar to a given embedding.

        Uses cosine similarity. Returns tracks with their similarity scores.

        Note: For large databases, consider using Qdrant for vector search
        and SQLite only for relational data.
        """
        query_vec = np.array(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        results = []
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM face_tracks")

            for row in cursor.fetchall():
                track_vec = np.frombuffer(
                    row["avg_embedding"], dtype=np.float32
                )
                track_norm = np.linalg.norm(track_vec)
                if track_norm == 0:
                    continue

                similarity = float(
                    np.dot(query_vec, track_vec) / (query_norm * track_norm)
                )
                if similarity >= threshold:
                    results.append((self._row_to_face_track(row), similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
