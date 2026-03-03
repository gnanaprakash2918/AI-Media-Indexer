"""Processor protocols for Dependency Inversion Principle.

These protocols define the expected interfaces for ML and data processing
components, allowing the pipeline to depend on abstractions rather than
concrete model implementations.
"""

from pathlib import Path
from typing import Any, Protocol


class AudioTranscriberProtocol(Protocol):
    """Protocol for speech-to-text processing."""

    def process_audio(self, audio_path: str, video_path: str) -> list[dict[str, Any]]:
        """Transcribe audio and return segments with timestamps."""
        ...


class VoiceDiarizerProtocol(Protocol):
    """Protocol for speaker diarization and voice feature extraction."""

    def process_voice(self, audio_path: str, video_path: str) -> list[dict[str, Any]]:
        """Identify speakers and extract embedding features."""
        ...


class VisualEncoderProtocol(Protocol):
    """Protocol for extracting visual embeddings from frames."""

    def encode_batch(self, images: list[Any]) -> Any:
        """Extract vector embeddings from a batch of images."""
        ...


class FrameAnalyzerProtocol(Protocol):
    """Protocol for visual analysis of video frames."""

    def analyze_frame(self, image: Any) -> dict[str, Any]:
        """Detect objects, text, or concepts in a single frame."""
        ...


class SceneDetectorProtocol(Protocol):
    """Protocol for detecting scene boundaries in video."""

    def detect_scenes(self, video_path: str) -> list[tuple[float, float]]:
        """Detect scene start and end times."""
        ...


class VLMCaptionerProtocol(Protocol):
    """Protocol for generating natural language descriptions using VLMs."""

    def generate_description(self, image_paths: list[str], prompt: str) -> str:
        """Generate a description based on images and text prompt."""
        ...
