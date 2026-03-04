"""TemporalEvent sequence operations for GraphRAG.

Extracted from IdentityGraphManager to reduce God class size.
"""

from __future__ import annotations

import sqlite3
import uuid
from threading import Lock
from typing import TYPE_CHECKING, Any


from core.storage.identity_models import (
    TemporalEvent,
)

if TYPE_CHECKING:
    pass


class TemporalEventManager:
    """TemporalEvent sequence operations for GraphRAG."""

    # These will be available via IdentityGraphManager inheritance
    db_path: str
    _lock: Lock
    # Type stub for type checkers
    def __getattr__(self, name: str) -> Any: ...

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

