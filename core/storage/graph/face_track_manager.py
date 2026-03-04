"""FaceTrack CRUD operations.

Extracted from IdentityGraphManager to reduce God class size.
"""

from __future__ import annotations

import sqlite3
import uuid
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from core.storage.identity_models import (
    FaceTrack,
)

if TYPE_CHECKING:
    pass


class FaceTrackManager:
    """FaceTrack CRUD operations."""

    # These will be available via IdentityGraphManager inheritance
    db_path: str
    _lock: Lock
    # Type stub for type checkers
    def __getattr__(self, name: str) -> Any: ...

    def create_face_track(
        self,
        media_id: str,
        start_frame: int,
        end_frame: int,
        start_time: float,
        end_time: float,
        avg_embedding: list[float],
        identity_id: str | None = None,
        best_thumbnail_path: str | None = None,
        avg_confidence: float = 0.0,
        frame_count: int = 1,
    ) -> FaceTrack:
        """Create a new face track."""
        track_id = str(uuid.uuid4())
        embedding_blob = np.array(avg_embedding, dtype=np.float32).tobytes()

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO face_tracks
                (id, media_id, start_frame, end_frame, start_time, end_time,
                 avg_embedding, identity_id, best_thumbnail_path, avg_confidence, frame_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    track_id,
                    media_id,
                    start_frame,
                    end_frame,
                    start_time,
                    end_time,
                    embedding_blob,
                    identity_id,
                    best_thumbnail_path,
                    avg_confidence,
                    frame_count,
                ),
            )
            conn.commit()

        return FaceTrack(
            id=track_id,
            media_id=media_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_time,
            end_time=end_time,
            avg_embedding=avg_embedding,
            identity_id=identity_id,
            best_thumbnail_path=best_thumbnail_path,
            avg_confidence=avg_confidence,
            frame_count=frame_count,
        )

    def get_face_tracks_for_media(self, media_id: str) -> list[FaceTrack]:
        """Get all face tracks for a specific media file."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM face_tracks WHERE media_id = ? ORDER BY start_time",
                (media_id,),
            )
            return [self._row_to_face_track(row) for row in cursor.fetchall()]

    def get_face_tracks_for_identity(self, identity_id: str) -> list[FaceTrack]:
        """Get all face tracks linked to an identity."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM face_tracks WHERE identity_id = ? ORDER BY created_at",
                (identity_id,),
            )
            return [self._row_to_face_track(row) for row in cursor.fetchall()]

    def get_unlinked_face_tracks(self, limit: int = 100) -> list[FaceTrack]:
        """Get face tracks without an identity (for HITL assignment)."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM face_tracks WHERE identity_id IS NULL ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [self._row_to_face_track(row) for row in cursor.fetchall()]

    def link_face_track_to_identity(
        self, track_id: str, identity_id: str
    ) -> bool:
        """Link a face track to an identity."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE face_tracks SET identity_id = ? WHERE id = ?",
                (identity_id, track_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_face_tracks_for_media(self, media_id: str) -> int:
        """Delete all face tracks for a media file (cascade on delete)."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM face_tracks WHERE media_id = ?", (media_id,)
            )
            conn.commit()
            return cursor.rowcount

    def _row_to_face_track(self, row: sqlite3.Row) -> FaceTrack:
        """Convert a database row to a FaceTrack object."""
        embedding = np.frombuffer(
            row["avg_embedding"], dtype=np.float32
        ).tolist()
        return FaceTrack(
            id=row["id"],
            media_id=row["media_id"],
            start_frame=row["start_frame"],
            end_frame=row["end_frame"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            avg_embedding=embedding,
            identity_id=row["identity_id"],
            best_thumbnail_path=row["best_thumbnail_path"],
            avg_confidence=row["avg_confidence"],
            frame_count=row["frame_count"],
        )

