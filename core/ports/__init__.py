"""Core protocols and interfaces for Dependency Inversion Principle.

Defines runtime-checkable Protocol classes enabling SOLID compliance across the system.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class EmbeddingProvider(Protocol):
    def encode_texts(
        self,
        texts: str | list[str],
        *,
        batch_size: int = 1,
        is_query: bool = False,
        job_id: str | None = None,
    ) -> list[list[float]]: ...

    def get_embedding(self, text: str) -> list[float]: ...


@runtime_checkable
class SearchProvider(Protocol):
    async def search(
        self,
        query: str,
        limit: int = 20,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


@runtime_checkable
class MediaProcessor(Protocol):
    async def process(self, path: Any, *, job_id: str | None = None) -> None: ...


@runtime_checkable
class AudioProcessor(Protocol):
    def transcribe(self, path: str) -> dict[str, Any]: ...


@runtime_checkable
class VoiceProcessor(Protocol):
    def process(self, path: str) -> dict[str, Any]: ...


@runtime_checkable
class VisionAnalyzer(Protocol):
    def analyze_frame(self, frame) -> dict[str, Any]: ...


@runtime_checkable
class SceneDetector(Protocol):
    def detect_scenes(self, video_path: str) -> list[tuple[float, float]]: ...


@runtime_checkable
class VLMProcessor(Protocol):
    def caption_scene(self, video_path: str, start: float, end: float) -> str: ...


@runtime_checkable
class FaceTracker(Protocol):
    def track_faces(self, video_path: str) -> list[dict[str, Any]]: ...


@runtime_checkable
class FrameRepository(Protocol):
    def upsert_media_frames_batch(self, frames: list[dict[str, Any]]) -> int: ...

    def get_frames_by_video(
        self,
        video_path: str,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class SegmentRepository(Protocol):
    def insert_media_segments(
        self,
        video_path: str,
        segments: list[dict[str, Any]],
        job_id: str | None = None,
    ) -> int: ...


@runtime_checkable
class AudioEventRepository(Protocol):
    def insert_audio_event(
        self,
        media_path: str,
        event_type: str,
        start_time: float,
        end_time: float,
        confidence: float,
        clap_embedding: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None: ...


@runtime_checkable
class IdentityGraphRepository(Protocol):
    def get_face_cluster_by_name(self, name: str) -> int | None: ...
    def get_speaker_cluster_by_name(self, name: str) -> int | None: ...


@runtime_checkable
class StorageBackend(
    FrameRepository,
    SegmentRepository,
    AudioEventRepository,
    IdentityGraphRepository,
    Protocol,
):
    def close(self) -> None: ...

    def update_media_metadata(
        self, media_path: str, metadata: dict[str, Any]
    ) -> None: ...
