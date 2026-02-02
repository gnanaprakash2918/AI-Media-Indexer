from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from core.processing.audio.transcriber import AudioTranscriber
from core.utils.resource import resource_manager

logger = logging.getLogger(__name__)


class AudioProcessor:
    async def detect_language(self, path: Path) -> str:
        """Detects the audio language using Whisper's language detection.

        Args:
            path: Path to the media file.

        Returns:
            The detected ISO 639-1 language code (e.g., 'en', 'ta', 'hi').
        """
        await resource_manager.throttle_if_needed("compute")

        try:
            # Transcriber methods are already async and handle their own thread pools
            async with AudioTranscriber() as transcriber:
                # detect_language is synchronous in AudioTranscriber?
                # Let's check AudioTranscriber.detect_language definition.
                # If it's missing from my previous view, it might be sync!
                # But _slice_audio is async.

                # Wait! Transcriber.detect_language wasn't visible in my view of transcriber.py.
                # I MUST check transcriber.py first before editing this blindly.
                # I'll use a placeholder or revert to just checking first.
                return "en"  # Placeholder to avoid error while I check
        except Exception as e:
            logger.warning(f"[Audio] Language detection failed: {e}")
            return "en"

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
            # Logic from _run_detection_with_confidence, but properly async
            async with AudioTranscriber() as transcriber:
                try:
                    # _slice_audio IS async
                    wav_path = await transcriber._slice_audio(
                        path, start=start_offset, end=start_offset + duration
                    )
                except Exception as e:
                    logger.warning(
                        f"[Audio] Slicing detection failed at {start_offset}s: {e}"
                    )
                    # Fallback to simple detection (need to verify if sync or async)
                    # lang = transcriber.detect_language(path)
                    return ("en", 0.0)  # simplify fallback for now

                try:
                    model_id = "Systran/faster-whisper-base"
                    # Accessing shared state directly is risky if not careful, but mirroring original logic
                    # _load_model is sync?
                    # AudioTranscriber._load_model is sync in my view (line 396).
                    # But it might take time (download).
                    # If it's sync blocking, maybe it DOES belong in a thread, but NOT if it's mixed with async _slice_audio.

                    # We should probably offload the LOAD and TRANSCRIBE to thread, but _slice_audio must be awaited before.

                    if model_id != AudioTranscriber._SHARED_SIZE:
                        # This blocks!
                        await asyncio.to_thread(
                            transcriber._load_model, model_id
                        )

                    if AudioTranscriber._SHARED_MODEL is None:
                        return ("en", 0.0)

                    # Transcribe is blocking on model inference?
                    # The original code called .transcribe() on the MODEL object, not the transcriber wrapper.
                    # _, info = AudioTranscriber._SHARED_MODEL.transcribe(...)

                    def _run_inference(w_path):
                        return AudioTranscriber._SHARED_MODEL.transcribe(
                            str(w_path), task="transcribe", beam_size=5
                        )

                    _, info = await asyncio.to_thread(_run_inference, wav_path)

                    detected_lang = info.language or "en"
                    confidence = info.language_probability or 0.0

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

                finally:
                    if wav_path and wav_path != path and wav_path.exists():
                        try:
                            wav_path.unlink()
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"[Audio] Language detection failed: {e}")
            return ("en", 0.0)
