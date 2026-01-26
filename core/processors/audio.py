"""Audio processing module for the media ingestion pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
import logging

from core.utils.resource import resource_manager
from core.utils.observe import observe
from core.ingestion.transcriber import AudioTranscriber

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio extraction, language detection, and transcription."""

    def __init__(self, db):
        self.db = db

    async def detect_language(self, path: Path) -> str:
        """Detects the audio language using Whisper's language detection.

        Args:
            path: Path to the media file.

        Returns:
            The detected ISO 639-1 language code (e.g., 'en', 'ta', 'hi').
        """
        await resource_manager.throttle_if_needed("compute")

        try:
            return await asyncio.to_thread(self._run_detection, path)
        except Exception as e:
            logger.warning(f"[Audio] Language detection failed: {e}")
            return "en"

    def _run_detection(self, path: Path) -> str:
        """Synchronous helper for language detection."""
        with AudioTranscriber() as transcriber:
            return transcriber.detect_language(path)

    async def detect_language_with_confidence(
        self,
        path: Path,
        start_offset: float = 0.0,
        duration: float = 30.0,
    ) -> tuple[str, float]:
        """Detects audio language with confidence score for multi-pass detection.

        Args:
            path: Path to the media file.
            start_offset: Start position in seconds for audio sampling.
            duration: Duration in seconds to sample for detection.

        Returns:
            Tuple of (language_code, confidence_score).
        """
        await resource_manager.throttle_if_needed("compute")

        try:
            return await asyncio.to_thread(
                self._run_detection_with_confidence,
                path,
                start_offset,
                duration,
            )
        except Exception as e:
            logger.warning(f"[Audio] Language detection failed: {e}")
            return ("en", 0.0)

    def _run_detection_with_confidence(
        self,
        path: Path,
        start_offset: float,
        duration: float,
    ) -> tuple[str, float]:
        """Synchronous helper for language detection with confidence."""
        with AudioTranscriber() as transcriber:
            try:
                wav_path = transcriber._slice_audio(
                    path, start=start_offset, end=start_offset + duration
                )
            except Exception as e:
                logger.warning(
                    f"[Audio] Slicing detection failed at {start_offset}s: {e}"
                )
                lang = transcriber.detect_language(path)
                return (lang, 0.5)

            try:
                model_id = "Systran/faster-whisper-base"
                if AudioTranscriber._SHARED_SIZE != model_id:
                    transcriber._load_model(model_id)

                if AudioTranscriber._SHARED_MODEL is None:
                    return ("en", 0.0)

                _, info = AudioTranscriber._SHARED_MODEL.transcribe(
                    str(wav_path), task="transcribe", beam_size=5
                )

                detected_lang = info.language or "en"
                confidence = info.language_probability or 0.0

                indic_langs = ["ta", "hi", "te", "ml", "kn", "bn", "gu", "mr", "or", "pa"]
                if detected_lang in indic_langs and confidence > 0.2:
                    confidence = min(confidence * 1.5, 0.95)

                return (detected_lang, confidence)

            finally:
                if wav_path and wav_path != path and wav_path.exists():
                    try:
                        wav_path.unlink()
                    except Exception:
                        pass
