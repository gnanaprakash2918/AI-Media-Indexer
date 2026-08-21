"""Audio language detection utilities for the ingestion pipeline.

Extracted from IngestionPipeline to isolate language detection logic
from the main orchestration.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

from core.processing.transcriber import AudioTranscriber
from core.utils.logger import log
from core.utils.hardware import RESOURCE_ARBITER


async def detect_audio_language(path: Path) -> str:
    """Detect audio language using Whisper's language detection.

    Args:
        path: Path to the media file.

    Returns:
        ISO 639-1 language code (e.g., 'en', 'ta', 'hi').
    """

    try:
        return await asyncio.to_thread(_run_detection_sync, path)
    except Exception as e:
        log(f"[Audio] Language detection failed: {e}")
        return "en"


def _run_detection_sync(path: Path) -> str:
    """Synchronous language detection helper."""
    with AudioTranscriber() as transcriber:
        return transcriber.detect_language(path)


async def detect_audio_language_with_confidence(
    path: Path,
    start_offset: float = 0.0,
    duration: float = 30.0,
) -> tuple[str, float]:
    """Detect audio language with confidence score for multi-pass detection.

    Args:
        path: Path to the media file.
        start_offset: Start position in seconds for audio sampling.
        duration: Duration in seconds to sample for detection.

    Returns:
        Tuple of (language_code, confidence_score).
    """

    wav_path = None
    try:
        # Slice audio segment for detection
        try:
            with AudioTranscriber() as transcriber:
                wav_path = await transcriber._slice_audio(
                    path, start=start_offset, end=start_offset + duration
                )
        except Exception as e:
            log(f"[Audio] Slicing failed: {e}, falling back to full file")
            wav_path = path

        return await asyncio.to_thread(
            _run_detection_with_confidence_sync, wav_path
        )

    except Exception as e:
        log(f"[Audio] Language detection failed: {e}")
        return ("en", 0.0)

    finally:
        if (
            wav_path
            and isinstance(wav_path, Path)
            and wav_path != path
            and wav_path.exists()
        ):
            try:
                wav_path.unlink()
            except Exception:
                pass


def _run_detection_with_confidence_sync(
    wav_input: Path | bytes,
) -> tuple[str, float]:
    """Synchronous language detection with confidence scoring."""
    with AudioTranscriber() as transcriber:
        try:
            model_id = "Systran/faster-whisper-base"
            if model_id != AudioTranscriber._SHARED_SIZE:
                transcriber._load_model(model_id)

            if AudioTranscriber._SHARED_MODEL is None:
                return ("en", 0.0)

            if isinstance(wav_input, bytes):
                input_file = io.BytesIO(wav_input)
            else:
                input_file = str(wav_input)

            _, info = AudioTranscriber._SHARED_MODEL.transcribe(
                input_file,
                task="transcribe",
                beam_size=5,
            )

            detected_lang = info.language or "en"
            confidence = info.language_probability or 0.0

            # Boost confidence for Indic languages (Whisper often underestimates)
            indic_langs = [
                "ta",
                "hi",
                "te",
                "ml",
                "kn",
                "bn",
                "gu",
                "mr",
                "or",
                "pa",
            ]
            if detected_lang in indic_langs and confidence > 0.2:
                confidence = min(confidence * 1.5, 0.95)

            return (detected_lang, confidence)
        except Exception as e:
            log(f"[Audio] Detection inner error: {e}")
            return ("en", 0.0)
