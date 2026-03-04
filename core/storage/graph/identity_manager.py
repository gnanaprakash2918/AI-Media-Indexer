"""Identity CRUD and linking operations.

Extracted from IdentityGraphManager to reduce God class size.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from threading import Lock
from typing import TYPE_CHECKING, Any


from core.storage.identity_models import (
    Identity,
)
from core.utils.logger import log

if TYPE_CHECKING:
    pass


class IdentityManager:
    """Identity CRUD and linking operations."""

    # These will be available via IdentityGraphManager inheritance
    db_path: str
    _lock: Lock
    # Type stub for type checkers
    def __getattr__(self, name: str) -> Any: ...

    def create_identity(
        self, name: str | None = None, is_verified: bool = False
    ) -> Identity:
        """Create a new identity (person).

        Args:
            name: Optional name for the person.
            is_verified: Whether the identity has been verified by HITL.

        Returns:
            The created Identity object.
        """
        identity_id = str(uuid.uuid4())
        now = time.time()

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO identities (id, name, is_verified, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (identity_id, name, int(is_verified), now, now),
            )
            conn.commit()

        log(f"Created identity: {identity_id} (name={name})")
        return Identity(
            id=identity_id, name=name, is_verified=is_verified, created_at=now
        )

    def get_identity(self, identity_id: str) -> Identity | None:
        """Get an identity by ID."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM identities WHERE id = ?", (identity_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_identity(row)

    def get_identity_by_name(self, name: str) -> Identity | None:
        """Get an identity by name (case-insensitive)."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM identities WHERE LOWER(name) = LOWER(?)", (name,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_identity(row)

    def get_or_create_identity_by_name(self, name: str) -> Identity:
        """Get existing identity by name or create a new one."""
        identity = self.get_identity_by_name(name)
        if identity:
            return identity
        return self.create_identity(name=name, is_verified=True)

    def get_all_identities(self, limit: int = 100) -> list[Identity]:
        """Get all identities, enriched with track counts."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    i.*,
                    (SELECT COUNT(*) FROM face_tracks WHERE identity_id = i.id) as face_count,
                    (SELECT COUNT(*) FROM voice_tracks WHERE identity_id = i.id) as voice_count
                FROM identities i
                ORDER BY i.created_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            results = []
            for row in cursor.fetchall():
                identity = self._row_to_identity(row)
                identity.face_track_count = row["face_count"]
                identity.voice_track_count = row["voice_count"]
                results.append(identity)
            return results

    def update_identity_name(self, identity_id: str, name: str) -> bool:
        """Update the name of an identity (HITL naming)."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE identities SET name = ?, is_verified = 1, updated_at = ? WHERE id = ?",
                (name, time.time(), identity_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def link_faces_to_identity(
        self, face_ids: list[str], identity_id: str
    ) -> int:
        """Link multiple face tracks (by ID) to an identity.

        Note: Matches are done on ID, but face_ids from Qdrant might
        correspond to `FaceTrack` IDs if they share UUIDs, OR we match
        via embedding lookup. Here we assume strict ID match.
        """
        with self._lock, sqlite3.connect(self.db_path) as conn:
            # We use 'IN' clause with placeholders
            placeholders = ",".join(["?"] * len(face_ids))
            sql = f"UPDATE face_tracks SET identity_id = ? WHERE id IN ({placeholders})"
            cursor = conn.execute(sql, (identity_id, *face_ids))
            conn.commit()
            return cursor.rowcount

    def link_voices_to_identity(
        self, voice_ids: list[str], identity_id: str
    ) -> int:
        """Link multiple voice tracks (by ID) to an identity."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join(["?"] * len(voice_ids))
            sql = f"UPDATE voice_tracks SET identity_id = ? WHERE id IN ({placeholders})"
            cursor = conn.execute(sql, (identity_id, *voice_ids))
            conn.commit()
            return cursor.rowcount

    def merge_identities(self, from_id: str, to_id: str) -> int:
        """Merge one identity into another (HITL merge operation).

        All face/voice tracks from `from_id` are moved to `to_id`,
        then `from_id` is deleted.

        Args:
            from_id: Source identity to merge from.
            to_id: Target identity to merge into.

        Returns:
            Number of tracks moved.
        """
        moved = 0
        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Move face tracks
            cursor = conn.execute(
                "UPDATE face_tracks SET identity_id = ? WHERE identity_id = ?",
                (to_id, from_id),
            )
            moved += cursor.rowcount

            # Move voice tracks
            cursor = conn.execute(
                "UPDATE voice_tracks SET identity_id = ? WHERE identity_id = ?",
                (to_id, from_id),
            )
            moved += cursor.rowcount

            # Delete source identity
            conn.execute("DELETE FROM identities WHERE id = ?", (from_id,))
            conn.commit()

        log(f"Merged identity {from_id} -> {to_id}, moved {moved} tracks")
        return moved

    def delete_identity(self, identity_id: str) -> bool:
        """Delete an identity. Tracks are unlinked (not deleted)."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Unlink tracks first (FK will SET NULL, but explicit is clearer)
            conn.execute(
                "UPDATE face_tracks SET identity_id = NULL WHERE identity_id = ?",
                (identity_id,),
            )
            conn.execute(
                "UPDATE voice_tracks SET identity_id = NULL WHERE identity_id = ?",
                (identity_id,),
            )
            cursor = conn.execute(
                "DELETE FROM identities WHERE id = ?", (identity_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_identity(self, row: sqlite3.Row) -> Identity:
        """Convert a database row to an Identity object."""
        return Identity(
            id=row["id"],
            name=row["name"],
            is_verified=bool(row["is_verified"]),
            created_at=row["created_at"],
        )

