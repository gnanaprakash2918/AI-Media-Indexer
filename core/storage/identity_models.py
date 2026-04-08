"""Domain models for the Identity Graph system.

These dataclasses define the core entities used by IdentityGraphManager
for tracking people and temporal events across media files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


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
