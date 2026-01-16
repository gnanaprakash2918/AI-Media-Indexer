"""Audio Content Classifier using inaSpeechSegmenter.

This module provides speech/music/silence classification to route audio
segments to appropriate transcription modes:
- SPEECH → Normal Whisper with VAD
- MUSIC → force_lyrics mode (disabled VAD, low thresholds)
- NONENERGY → Skip (prevents hallucination)

Reference: MIREX 2018 Speech Detection Challenge Winner
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.utils.logger import log


class ContentType(Enum):
    """Audio content classification types."""

    SPEECH = "speech"
    MUSIC = "music"
    SILENCE = "silence"
    NOISE = "noise"


@dataclass
class AudioRegion:
    """Classified audio region with start/end timestamps."""

    content_type: ContentType
    start: float
    end: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        """Get region duration in seconds."""
        return self.end - self.start


class ContentClassifier:
    """Classifies audio content into speech/music/silence regions.

    Uses inaSpeechSegmenter for CNN-based classification, with fallback
    to simple energy-based detection if unavailable.
    """

    _instance = None
    _segmenter = None

    def __new__(cls):
        """Singleton pattern for model reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize classifier (lazy model loading)."""
        self._initialized = getattr(self, "_initialized", False)
        if self._initialized:
            return
        self._initialized = True
        self._use_ina = self._check_ina_available()

    def _check_ina_available(self) -> bool:
        """Check if inaSpeechSegmenter is installed."""
        try:
            import inaSpeechSegmenter  # noqa: F401

            log("[ContentClassifier] inaSpeechSegmenter available")
            return True
        except ImportError:
            log(
                "[ContentClassifier] inaSpeechSegmenter not installed, using fallback"
            )
            return False

    def _load_segmenter(self):
        """Lazy load the segmenter model."""
        if ContentClassifier._segmenter is not None:
            return ContentClassifier._segmenter

        if not self._use_ina:
            return None

        try:
            from inaSpeechSegmenter import Segmenter

            # Suppress TensorFlow warnings
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

            log("[ContentClassifier] Loading inaSpeechSegmenter model...")
            ContentClassifier._segmenter = Segmenter()
            log("[ContentClassifier] Model loaded successfully")
            return ContentClassifier._segmenter
        except Exception as e:
            log(f"[ContentClassifier] Failed to load model: {e}")
            self._use_ina = False
            return None

    def classify(self, audio_path: Path) -> list[AudioRegion]:
        """Classify audio file into content regions.

        Args:
            audio_path: Path to audio/video file

        Returns:
            List of AudioRegion with content type and timestamps
        """
        if self._use_ina:
            return self._classify_with_ina(audio_path)
        else:
            return self._classify_fallback(audio_path)

    def _classify_with_ina(self, audio_path: Path) -> list[AudioRegion]:
        """Classify using inaSpeechSegmenter (SOTA approach).

        inaSpeechSegmenter returns labels:
        - 'speech': Male or female speech
        - 'music': Music or singing
        - 'noEnergy': Silence
        - 'noise': Background noise
        """
        segmenter = self._load_segmenter()
        if segmenter is None:
            return self._classify_fallback(audio_path)

        try:
            # Run classification
            segments = segmenter(str(audio_path))

            regions = []
            for label, start, end in segments:
                # Map inaSpeechSegmenter labels to ContentType
                if label in ("male", "female", "speech"):
                    content_type = ContentType.SPEECH
                elif label == "music":
                    content_type = ContentType.MUSIC
                elif label == "noEnergy":
                    content_type = ContentType.SILENCE
                else:
                    content_type = ContentType.NOISE

                regions.append(
                    AudioRegion(
                        content_type=content_type,
                        start=float(start),
                        end=float(end),
                    )
                )

            # Log summary
            speech_dur = sum(
                r.duration
                for r in regions
                if r.content_type == ContentType.SPEECH
            )
            music_dur = sum(
                r.duration
                for r in regions
                if r.content_type == ContentType.MUSIC
            )
            silence_dur = sum(
                r.duration
                for r in regions
                if r.content_type == ContentType.SILENCE
            )
            total = speech_dur + music_dur + silence_dur

            if total > 0:
                log(
                    f"[ContentClassifier] {len(regions)} regions: "
                    f"{speech_dur:.1f}s speech ({100 * speech_dur / total:.0f}%), "
                    f"{music_dur:.1f}s music ({100 * music_dur / total:.0f}%), "
                    f"{silence_dur:.1f}s silence ({100 * silence_dur / total:.0f}%)"
                )

            return regions

        except Exception as e:
            log(f"[ContentClassifier] Classification failed: {e}")
            return self._classify_fallback(audio_path)

    def _classify_fallback(self, audio_path: Path) -> list[AudioRegion]:
        """Fallback: treat entire file as potential speech.

        Without inaSpeechSegmenter, we can't distinguish content types.
        This allows normal Whisper VAD to handle segmentation.
        """
        try:
            import json
            import subprocess

            # Use ffprobe to get duration
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(audio_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            info = json.loads(result.stdout)
            duration = float(info.get("format", {}).get("duration", 0))

            if duration > 0:
                log(
                    f"[ContentClassifier] Fallback mode: treating {duration:.1f}s as potential speech"
                )
                return [
                    AudioRegion(
                        content_type=ContentType.SPEECH,
                        start=0.0,
                        end=duration,
                        confidence=0.5,  # Low confidence = fallback
                    )
                ]
        except Exception:
            pass

        return []

    def get_speech_regions(
        self, regions: list[AudioRegion]
    ) -> list[AudioRegion]:
        """Filter to only speech regions."""
        return [r for r in regions if r.content_type == ContentType.SPEECH]

    def get_music_regions(
        self, regions: list[AudioRegion]
    ) -> list[AudioRegion]:
        """Filter to only music regions."""
        return [r for r in regions if r.content_type == ContentType.MUSIC]

    def should_use_lyrics_mode(self, regions: list[AudioRegion]) -> bool:
        """Check if audio is predominantly music (needs lyrics mode)."""
        total = sum(r.duration for r in regions)
        if total == 0:
            return False

        music_dur = sum(
            r.duration for r in regions if r.content_type == ContentType.MUSIC
        )
        speech_dur = sum(
            r.duration for r in regions if r.content_type == ContentType.SPEECH
        )

        # Use lyrics mode if >60% music or music > 2x speech
        return (music_dur / total > 0.6) or (music_dur > 2 * speech_dur)


# Singleton accessor
def get_content_classifier() -> ContentClassifier:
    """Get or create the content classifier instance."""
    return ContentClassifier()
