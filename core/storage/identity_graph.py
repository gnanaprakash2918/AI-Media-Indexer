"""Identity Graph Manager for tracking people across media.

This module implements a relational Identity Graph structure:
- Identity: A named person (e.g., "Prakash")
- FaceTrack: A sequence of face detections within a single video
- VoiceTrack: A sequence of voice segments within a single video

The key insight: Cluster faces/voices into Tracks WITHIN a video first,
then link Tracks to global Identities. This prevents bad clustering from
"Prakash in dark room" vs "Prakash outside" being treated as different people.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock

import numpy as np

from core.utils.logger import log


class TrackType(str, Enum):
    """Type of identity track."""

    FACE = "face"
    VOICE = "voice"


@dataclass
class Identity:
    """A named person in the system."""

    id: str
    name: str | None = None
    is_verified: bool = False
    created_at: float = field(default_factory=time.time)

    # Aggregated stats
    face_track_count: int = 0
    voice_track_count: int = 0
    total_appearances: int = 0


@dataclass
class FaceTrack:
    """A sequence of face detections within a single video.

    Temporal continuity: Faces are grouped if they appear in consecutive
    frames with high IoU overlap and embedding similarity.
    """

    id: str
    media_id: str  # video_path or hash
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    avg_embedding: list[float]  # Average of all face embeddings in track
    identity_id: str | None = None

    # Quality metrics
    best_thumbnail_path: str | None = None
    avg_confidence: float = 0.0
    frame_count: int = 0


@dataclass
class VoiceTrack:
    """A sequence of voice segments within a single video."""

    id: str
    media_id: str
    start_time: float
    end_time: float
    embedding: list[float]
    identity_id: str | None = None

    # Metadata
    speaker_label: str | None = None
    total_duration: float = 0.0


@dataclass
class Scene:
    """A narrative scene within a video for GraphRAG.

    Scenes are detected by shot boundaries plus semantic clustering.
    They form the nodes in the temporal graph.
    """

    id: str
    media_id: str
    start_time: float
    end_time: float
    location: str | None = None
    description: str | None = None
    scene_type: str | None = None  # "dialogue", "action", "transition", etc.

    # Linked identities in this scene
    face_cluster_ids: list[int] = field(default_factory=list)
    speaker_cluster_ids: list[int] = field(default_factory=list)

    # Detected entities and actions
    entities: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass
class SceneTransition:
    """An edge between two scenes in the temporal graph.

    Types: CUT, FADE, DISSOLVE, or semantic transitions.
    """

    id: str
    from_scene_id: str
    to_scene_id: str
    transition_type: str  # "cut", "fade", "dissolve", "wipe", "semantic"
    confidence: float = 1.0


@dataclass
class TemporalEvent:
    """A discrete event within a scene for fine-grained temporal queries.

    Events are the atoms of temporal reasoning:
    "Prakash speaks" -> "Alia responds" -> "Prakash walks away"
    """

    id: str
    scene_id: str
    identity_id: str | None
    timestamp: float
    event_type: str  # "speaks", "enters", "exits", "action", "gesture"
    description: str | None = None

    # For sequence queries
    previous_event_id: str | None = None
    next_event_id: str | None = None


class IdentityGraphManager:
    """SQLite-backed Identity Graph for robust person tracking.

    Key features:
    - Track-level clustering (within video) before global identity linking
    - Merge/split identities via HITL
    - Crash-safe atomic operations
    """

    DB_PATH = "identity_graph.db"

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the Identity Graph manager.

        Args:
            db_path: Path to SQLite database. Defaults to project root.
        """
        self.db_path = db_path or self.DB_PATH
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite schema with all required tables."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # Crash-safe writes

            # Identity table: Named people
            conn.execute("""
                CREATE TABLE IF NOT EXISTS identities (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    is_verified INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (unixepoch()),
                    updated_at REAL DEFAULT (unixepoch()),
                    notes TEXT
                )
            """)

            # FaceTrack table: Face sequences within a video
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_tracks (
                    id TEXT PRIMARY KEY,
                    media_id TEXT NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    avg_embedding BLOB NOT NULL,
                    identity_id TEXT,
                    best_thumbnail_path TEXT,
                    avg_confidence REAL DEFAULT 0.0,
                    frame_count INTEGER DEFAULT 1,
                    created_at REAL DEFAULT (unixepoch()),
                    FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE SET NULL
                )
            """)

            # VoiceTrack table: Voice sequences within a video
            conn.execute("""
                CREATE TABLE IF NOT EXISTS voice_tracks (
                    id TEXT PRIMARY KEY,
                    media_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    embedding BLOB NOT NULL,
                    identity_id TEXT,
                    speaker_label TEXT,
                    total_duration REAL DEFAULT 0.0,
                    created_at REAL DEFAULT (unixepoch()),
                    FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE SET NULL
                )
            """)

            # Indexes for fast lookups
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_face_tracks_media ON face_tracks(media_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_face_tracks_identity ON face_tracks(identity_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_voice_tracks_media ON voice_tracks(media_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_voice_tracks_identity ON voice_tracks(identity_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_identities_name ON identities(name)"
            )

            # =========================================================
            # GRAPHRAG TABLES FOR TEMPORAL QUERIES
            # =========================================================

            # Scene table: Narrative scenes for temporal graph nodes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scenes (
                    id TEXT PRIMARY KEY,
                    media_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    location TEXT,
                    description TEXT,
                    scene_type TEXT,
                    face_cluster_ids TEXT,
                    speaker_cluster_ids TEXT,
                    entities TEXT,
                    actions TEXT,
                    created_at REAL DEFAULT (unixepoch())
                )
            """)

            # SceneTransition table: Edges between scenes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scene_transitions (
                    id TEXT PRIMARY KEY,
                    from_scene_id TEXT NOT NULL,
                    to_scene_id TEXT NOT NULL,
                    transition_type TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (from_scene_id) REFERENCES scenes(id) ON DELETE CASCADE,
                    FOREIGN KEY (to_scene_id) REFERENCES scenes(id) ON DELETE CASCADE
                )
            """)

            # TemporalEvent table: Fine-grained events for sequence queries
            conn.execute("""
                CREATE TABLE IF NOT EXISTS temporal_events (
                    id TEXT PRIMARY KEY,
                    scene_id TEXT NOT NULL,
                    identity_id TEXT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT,
                    previous_event_id TEXT,
                    next_event_id TEXT,
                    created_at REAL DEFAULT (unixepoch()),
                    FOREIGN KEY (scene_id) REFERENCES scenes(id) ON DELETE CASCADE,
                    FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE SET NULL
                )
            """)

            # GraphRAG indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenes_media ON scenes(media_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenes_time ON scenes(start_time, end_time)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scene_transitions_from ON scene_transitions(from_scene_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scene_transitions_to ON scene_transitions(to_scene_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_temporal_events_scene ON temporal_events(scene_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_temporal_events_identity ON temporal_events(identity_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_temporal_events_time ON temporal_events(timestamp)"
            )

            conn.commit()
            conn.commit()
            log("Identity Graph database initialized", db_path=self.db_path)

    def get_stats(self) -> dict[str, int]:
        """Get graph statistics."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT
                    (SELECT COUNT(*) FROM identities) as identities,
                    (SELECT COUNT(*) FROM face_tracks) as face_tracks,
                    (SELECT COUNT(*) FROM voice_tracks) as voice_tracks
            """).fetchone()
            return {
                "identities": row[0],
                "tracks": row[1] + row[2],
                "face_tracks": row[1],
                "voice_tracks": row[2],
            }

    # =========================================================================
    # IDENTITY OPERATIONS
    # =========================================================================

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

    # =========================================================================
    # FACE TRACK OPERATIONS
    # =========================================================================

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

    # =========================================================================
    # VOICE TRACK OPERATIONS
    # =========================================================================

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

    # =========================================================================
    # SEARCH & QUERY OPERATIONS
    # =========================================================================

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

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _row_to_identity(self, row: sqlite3.Row) -> Identity:
        """Convert a database row to an Identity object."""
        return Identity(
            id=row["id"],
            name=row["name"],
            is_verified=bool(row["is_verified"]),
            created_at=row["created_at"],
        )

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

    # =========================================================================
    # GRAPHRAG: SCENE OPERATIONS
    # =========================================================================

    def create_scene(
        self,
        media_id: str,
        start_time: float,
        end_time: float,
        location: str | None = None,
        description: str | None = None,
        scene_type: str | None = None,
        face_cluster_ids: list[int] | None = None,
        speaker_cluster_ids: list[int] | None = None,
        entities: list[str] | None = None,
        actions: list[str] | None = None,
    ) -> Scene:
        """Create a new scene in the temporal graph."""
        import json

        scene_id = str(uuid.uuid4())

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO scenes
                (id, media_id, start_time, end_time, location, description, 
                 scene_type, face_cluster_ids, speaker_cluster_ids, entities, actions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_id,
                    media_id,
                    start_time,
                    end_time,
                    location,
                    description,
                    scene_type,
                    json.dumps(face_cluster_ids or []),
                    json.dumps(speaker_cluster_ids or []),
                    json.dumps(entities or []),
                    json.dumps(actions or []),
                ),
            )
            conn.commit()

        return Scene(
            id=scene_id,
            media_id=media_id,
            start_time=start_time,
            end_time=end_time,
            location=location,
            description=description,
            scene_type=scene_type,
            face_cluster_ids=face_cluster_ids or [],
            speaker_cluster_ids=speaker_cluster_ids or [],
            entities=entities or [],
            actions=actions or [],
        )

    def get_scenes_for_media(self, media_id: str) -> list[Scene]:
        """Get all scenes for a video, ordered by time."""
        import json

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM scenes WHERE media_id = ? ORDER BY start_time",
                (media_id,),
            )
            scenes = []
            for row in cursor.fetchall():
                scenes.append(
                    Scene(
                        id=row["id"],
                        media_id=row["media_id"],
                        start_time=row["start_time"],
                        end_time=row["end_time"],
                        location=row["location"],
                        description=row["description"],
                        scene_type=row["scene_type"],
                        face_cluster_ids=json.loads(
                            row["face_cluster_ids"] or "[]"
                        ),
                        speaker_cluster_ids=json.loads(
                            row["speaker_cluster_ids"] or "[]"
                        ),
                        entities=json.loads(row["entities"] or "[]"),
                        actions=json.loads(row["actions"] or "[]"),
                    )
                )
            return scenes

    def create_scene_transition(
        self,
        from_scene_id: str,
        to_scene_id: str,
        transition_type: str = "cut",
        confidence: float = 1.0,
    ) -> SceneTransition:
        """Create a transition edge between two scenes."""
        transition_id = str(uuid.uuid4())

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO scene_transitions
                (id, from_scene_id, to_scene_id, transition_type, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    transition_id,
                    from_scene_id,
                    to_scene_id,
                    transition_type,
                    confidence,
                ),
            )
            conn.commit()

        return SceneTransition(
            id=transition_id,
            from_scene_id=from_scene_id,
            to_scene_id=to_scene_id,
            transition_type=transition_type,
            confidence=confidence,
        )

    def get_scene_timeline(self, media_id: str) -> list[dict]:
        """Get the complete scene timeline with transitions for a video.

        Returns ordered scenes with their transitions for narrative visualization.
        """
        scenes = self.get_scenes_for_media(media_id)
        if not scenes:
            return []

        # Get all transitions between these scenes
        scene_ids = [s.id for s in scenes]
        placeholders = ",".join(["?"] * len(scene_ids))

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"""
                SELECT * FROM scene_transitions 
                WHERE from_scene_id IN ({placeholders}) OR to_scene_id IN ({placeholders})
                """,
                scene_ids + scene_ids,
            )
            transitions = {
                row["from_scene_id"]: row for row in cursor.fetchall()
            }

        # Build timeline
        timeline = []
        for i, scene in enumerate(scenes):
            entry = {
                "scene": {
                    "id": scene.id,
                    "start_time": scene.start_time,
                    "end_time": scene.end_time,
                    "duration": scene.end_time - scene.start_time,
                    "location": scene.location,
                    "description": scene.description,
                    "scene_type": scene.scene_type,
                    "people": scene.face_cluster_ids,
                    "actions": scene.actions,
                },
                "transition_to_next": None,
            }

            if scene.id in transitions:
                t = transitions[scene.id]
                entry["transition_to_next"] = {
                    "type": t["transition_type"],
                    "confidence": t["confidence"],
                }

            timeline.append(entry)

        return timeline

    # =========================================================================
    # GRAPHRAG: TEMPORAL EVENT OPERATIONS
    # =========================================================================

    def create_temporal_event(
        self,
        scene_id: str,
        timestamp: float,
        event_type: str,
        identity_id: str | None = None,
        description: str | None = None,
    ) -> TemporalEvent:
        """Create a new temporal event within a scene."""
        event_id = str(uuid.uuid4())

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO temporal_events
                (id, scene_id, identity_id, timestamp, event_type, description)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    scene_id,
                    identity_id,
                    timestamp,
                    event_type,
                    description,
                ),
            )
            conn.commit()

        return TemporalEvent(
            id=event_id,
            scene_id=scene_id,
            identity_id=identity_id,
            timestamp=timestamp,
            event_type=event_type,
            description=description,
        )

    def link_events_sequence(self, event_ids: list[str]) -> int:
        """Link a sequence of events in order (A→B→C).

        Args:
            event_ids: List of event IDs in chronological order.

        Returns:
            Number of links created.
        """
        if len(event_ids) < 2:
            return 0

        links_created = 0
        with self._lock, sqlite3.connect(self.db_path) as conn:
            for i in range(len(event_ids) - 1):
                prev_id = event_ids[i]
                next_id = event_ids[i + 1]

                # Update forward link
                conn.execute(
                    "UPDATE temporal_events SET next_event_id = ? WHERE id = ?",
                    (next_id, prev_id),
                )
                # Update backward link
                conn.execute(
                    "UPDATE temporal_events SET previous_event_id = ? WHERE id = ?",
                    (prev_id, next_id),
                )
                links_created += 1

            conn.commit()

        return links_created

    def find_event_sequence(
        self,
        pattern: list[str],
        media_id: str | None = None,
        identity_name: str | None = None,
    ) -> list[list[dict]]:
        """Find sequences of events matching a pattern.

        This is the KEY GraphRAG query for temporal sequences like:
        "A speaks" → "B responds" → "A walks away"

        Args:
            pattern: List of event_type patterns to match in order.
                     e.g., ["speaks", "speaks", "exits"]
            media_id: Optional filter to specific video.
            identity_name: Optional filter to specific person.

        Returns:
            List of matching sequences, each containing event dicts.
        """
        if not pattern:
            return []

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build query for first event type
            conditions = ["event_type = ?"]
            params: list = [pattern[0]]

            if media_id:
                conditions.append(
                    "scene_id IN (SELECT id FROM scenes WHERE media_id = ?)"
                )
                params.append(media_id)

            if identity_name:
                conditions.append(
                    "identity_id IN (SELECT id FROM identities WHERE LOWER(name) = LOWER(?))"
                )
                params.append(identity_name)

            where_clause = " AND ".join(conditions)

            # Get all starting events
            cursor = conn.execute(
                f"SELECT * FROM temporal_events WHERE {where_clause} ORDER BY timestamp",
                params,
            )
            starting_events = cursor.fetchall()

            if not starting_events:
                return []

            # For each starting event, try to match the full pattern
            matched_sequences = []

            for start_event in starting_events:
                sequence = [dict(start_event)]
                current_event = start_event
                pattern_matched = True

                for i in range(1, len(pattern)):
                    # Get next event
                    next_id = current_event["next_event_id"]
                    if not next_id:
                        pattern_matched = False
                        break

                    next_event = conn.execute(
                        "SELECT * FROM temporal_events WHERE id = ?",
                        (next_id,),
                    ).fetchone()

                    if not next_event or next_event["event_type"] != pattern[i]:
                        pattern_matched = False
                        break

                    sequence.append(dict(next_event))
                    current_event = next_event

                if pattern_matched and len(sequence) == len(pattern):
                    matched_sequences.append(sequence)

            return matched_sequences

    def get_events_for_identity(
        self,
        identity_id: str,
        media_id: str | None = None,
        limit: int = 100,
    ) -> list[TemporalEvent]:
        """Get all events involving a specific identity."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if media_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM temporal_events 
                    WHERE identity_id = ?
                    AND scene_id IN (SELECT id FROM scenes WHERE media_id = ?)
                    ORDER BY timestamp
                    LIMIT ?
                    """,
                    (identity_id, media_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM temporal_events 
                    WHERE identity_id = ?
                    ORDER BY timestamp
                    LIMIT ?
                    """,
                    (identity_id, limit),
                )

            events = []
            for row in cursor.fetchall():
                events.append(
                    TemporalEvent(
                        id=row["id"],
                        scene_id=row["scene_id"],
                        identity_id=row["identity_id"],
                        timestamp=row["timestamp"],
                        event_type=row["event_type"],
                        description=row["description"],
                        previous_event_id=row["previous_event_id"],
                        next_event_id=row["next_event_id"],
                    )
                )
            return events

    def get_graphrag_stats(self) -> dict[str, int]:
        """Get GraphRAG statistics including scenes, transitions, events."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT
                    (SELECT COUNT(*) FROM identities) as identities,
                    (SELECT COUNT(*) FROM face_tracks) as face_tracks,
                    (SELECT COUNT(*) FROM voice_tracks) as voice_tracks,
                    (SELECT COUNT(*) FROM scenes) as scenes,
                    (SELECT COUNT(*) FROM scene_transitions) as transitions,
                    (SELECT COUNT(*) FROM temporal_events) as events
            """).fetchone()
            return {
                "identities": row[0],
                "face_tracks": row[1],
                "voice_tracks": row[2],
                "scenes": row[3],
                "scene_transitions": row[4],
                "temporal_events": row[5],
            }


# Global instance
identity_graph = IdentityGraphManager()
