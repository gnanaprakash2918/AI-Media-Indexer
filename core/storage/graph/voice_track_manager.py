"""VoiceTrack CRUD operations.

Extracted from IdentityGraphManager to reduce God class size.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from core.storage.identity_models import (
    FaceTrack,
    Identity,
    Scene,
    SceneTransition,
    TemporalEvent,
    TrackType,
    VoiceTrack,
)
from core.utils.logger import log

if TYPE_CHECKING:
    from core.storage.identity_graph import IdentityGraphManager


class VoiceTrackManager:
    """VoiceTrack CRUD operations."""

    # These will be available via IdentityGraphManager inheritance
    db_path: str
    _lock: Lock
    # Type stub for type checkers
    def __getattr__(self, name: str) -> Any: ...

    def create_voice_track(
        self,
        media_id: str,
        start_time: float,
        end_time: float,
        embedding: list[float],
        identity_id: str | None = None,
        speaker_label: str | None = None,
    ) -> VoiceTrack:
        """Create a new voice track."""
        track_id = str(uuid.uuid4())
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        total_duration = end_time - start_time

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO voice_tracks
                (id, media_id, start_time, end_time, embedding, identity_id, speaker_label, total_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    track_id,
                    media_id,
                    start_time,
                    end_time,
                    embedding_blob,
                    identity_id,
                    speaker_label,
                    total_duration,
                ),
            )
            conn.commit()

        return VoiceTrack(
            id=track_id,
            media_id=media_id,
            start_time=start_time,
            end_time=end_time,
            embedding=embedding,
            identity_id=identity_id,
            speaker_label=speaker_label,
            total_duration=total_duration,
        )

    def get_voice_tracks_for_media(self, media_id: str) -> list[VoiceTrack]:
        """Get all voice tracks for a specific media file."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM voice_tracks WHERE media_id = ? ORDER BY start_time",
                (media_id,),
            )
            return [self._row_to_voice_track(row) for row in cursor.fetchall()]

    def get_voice_tracks_for_identity(
        self, identity_id: str
    ) -> list[VoiceTrack]:
        """Get all voice tracks linked to an identity."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM voice_tracks WHERE identity_id = ? ORDER BY created_at",
                (identity_id,),
            )
            return [self._row_to_voice_track(row) for row in cursor.fetchall()]

    def link_voice_track_to_identity(
        self, track_id: str, identity_id: str
    ) -> bool:
        """Link a voice track to an identity."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE voice_tracks SET identity_id = ? WHERE id = ?",
                (identity_id, track_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_voice_tracks_for_media(self, media_id: str) -> int:
        """Delete all voice tracks for a media file."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM voice_tracks WHERE media_id = ?", (media_id,)
            )
            conn.commit()
            return cursor.rowcount

    def _row_to_voice_track(self, row: sqlite3.Row) -> VoiceTrack:
        """Convert a database row to a VoiceTrack object."""
        embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
        return VoiceTrack(
            id=row["id"],
            media_id=row["media_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            embedding=embedding,
            identity_id=row["identity_id"],
            speaker_label=row["speaker_label"],
            total_duration=row["total_duration"],
        )

