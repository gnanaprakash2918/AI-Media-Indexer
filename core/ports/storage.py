"""Storage repository protocols for Dependency Inversion Principle.

These protocols define the expected interfaces for storage operations,
allowing the ingestion pipeline and other components to depend on abstractions
rather than concrete database implementations like VectorDB.
"""

from typing import Any, Protocol


class FrameRepository(Protocol):
    """Protocol for frame storage and retrieval."""

    def upsert_media_frames_batch(self, frames: list[dict[str, Any]]) -> int:
        """Batch upsert multiple frame embeddings for better performance."""
        ...

    def get_frames_by_video(
        self,
        video_path: str,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve frame metadata for a video within optional time range."""
        ...


class SegmentRepository(Protocol):
    """Protocol for media segment storage and retrieval."""

    def insert_media_segments(
        self,
        video_path: str,
        segments: list[dict[str, Any]],
        job_id: str | None = None,
    ) -> int:
        """Insert media segments (dialogue, subtitles) into the database."""
        ...


class AudioEventRepository(Protocol):
    """Protocol for audio event storage."""

    def insert_audio_event(
        self,
        media_path: str,
        event_type: str,
        start_time: float,
        end_time: float,
        confidence: float,
        clap_embedding: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Insert audio event with optional CLAP embedding for semantic search."""
        ...


class IdentityGraphRepository(Protocol):
    """Protocol for identity graph operations."""

    def get_face_cluster_by_name(self, name: str) -> int | None:
        """Find face cluster ID by HITL-assigned name."""
        ...

    def get_speaker_cluster_by_name(self, name: str) -> int | None:
        """Find voice cluster ID by HITL-assigned speaker name."""
        ...


class StorageBackend(
    FrameRepository,
    SegmentRepository,
    AudioEventRepository,
    IdentityGraphRepository,
    Protocol,
):
    """Composite protocol representing the full storage backend contract.

    This represents the minimum viable interface that the pipeline needs.
    """

    def close(self) -> None:
        """Close the storage connection."""
        ...

    def update_media_metadata(self, media_path: str, metadata: dict[str, Any]) -> None:
        """Update video-level metadata."""
        ...
