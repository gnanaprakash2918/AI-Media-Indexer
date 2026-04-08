"""Mock implementations of processor protocols for isolated unit testing."""

from typing import Any

from core.ports.processors import (
    AudioProcessor,
    FaceTracker,
    VisionAnalyzer,
    VoiceProcessor,
)


class MockVisionAnalyzer(VisionAnalyzer):
    async def analyze_frame(self, image: Any) -> dict[str, Any]:
        return {
            "description": "Mock scene",
            "entities": [{"name": "mock entity"}],
            "actions": ["walking"],
            "visible_text": ["hello world"],
        }


class MockFaceTracker(FaceTracker):
    async def prepare_batches(
        self, video_path: str, frames_to_extract: list[dict]
    ) -> tuple[Any, int, list]:
        return [], 0, []

    async def get_face_batches(self) -> Any:
        return []

    async def end_extraction(self) -> None:
        pass


class MockAudioProcessor(AudioProcessor):
    def process_audio(self, video_path: str) -> dict[str, Any]:
        return {
            "transcription": [{"start": 0.0, "end": 2.0, "text": "hello"}],
            "language": "en",
        }


class MockVoiceProcessor(VoiceProcessor):
    async def process(self, video_path: str) -> list[dict]:
        return [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
