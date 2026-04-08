"""Scene and SceneTransition operations for GraphRAG.

Extracted from IdentityGraphManager to reduce God class size.
"""

from __future__ import annotations

import sqlite3
import uuid
from threading import Lock
from typing import TYPE_CHECKING, Any

from core.storage.identity_models import (
    Scene,
    SceneTransition,
)

if TYPE_CHECKING:
    pass


class SceneManager:
    """Scene and SceneTransition operations for GraphRAG."""

    # These will be available via IdentityGraphManager inheritance
    db_path: str
    _lock: Lock

    # Type stub for type checkers
    def __getattr__(self, name: str) -> Any: ...

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
