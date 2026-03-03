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


from core.storage.identity_models import (
    FaceTrack,
    Identity,
    Scene,
    SceneTransition,
    TemporalEvent,
    TrackType,
    VoiceTrack,
)


from core.storage.graph.identity_manager import IdentityManager
from core.storage.graph.face_track_manager import FaceTrackManager
from core.storage.graph.voice_track_manager import VoiceTrackManager
from core.storage.graph.scene_manager import SceneManager
from core.storage.graph.temporal_event_manager import TemporalEventManager
from core.storage.graph.query_manager import IdentityQueryManager

class IdentityGraphManager(
    IdentityManager,
    FaceTrackManager,
    VoiceTrackManager,
    SceneManager,
    TemporalEventManager,
    IdentityQueryManager,
):
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











    # =========================================================================
    # FACE TRACK OPERATIONS
    # =========================================================================







    # =========================================================================
    # VOICE TRACK OPERATIONS
    # =========================================================================






    # =========================================================================
    # SEARCH & QUERY OPERATIONS
    # =========================================================================



    # =========================================================================
    # HELPERS
    # =========================================================================




    # =========================================================================
    # GRAPHRAG: SCENE OPERATIONS
    # =========================================================================




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
