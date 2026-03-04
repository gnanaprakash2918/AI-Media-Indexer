"""Processor protocols for Dependency Inversion Principle.

These protocols define the expected interfaces for media processors,
allowing the ingestion pipeline to depend on abstractions rather than
concrete ML model implementations.
"""

from typing import Any, Protocol


class AudioProcessor(Protocol):
    """Protocol for audio transcription and analysis."""

    def transcribe(self, path: str) -> dict[str, Any]:
        """Transcribe audio and return segments with timestamps."""
        ...


class VoiceProcessor(Protocol):
    """Protocol for voice diarization and speaker identification."""

    def process(self, path: str) -> dict[str, Any]:
        """Process voice and return speaker segments."""
        ...


class VisionAnalyzer(Protocol):
    """Protocol for visual frame analysis (object/action detection)."""

    def analyze_frame(self, frame) -> dict[str, Any]:
        """Analyze a single frame for visual features."""
        ...


class SceneDetector(Protocol):
    """Protocol for video scene boundary detection."""

    def detect_scenes(self, video_path: str) -> list[tuple[float, float]]:
        """Detect scene boundaries and return (start, end) timestamps."""
        ...


class VLMProcessor(Protocol):
    """Protocol for Vision-Language Models (Video understanding)."""

    def caption_scene(self, video_path: str, start: float, end: float) -> str:
        """Generate a caption for a specific video scene."""
        ...


class FaceTracker(Protocol):
    """Protocol for face detection and tracking."""

    def track_faces(self, video_path: str) -> list[dict[str, Any]]:
        """Track faces across a video and return clustering data."""
        ...
