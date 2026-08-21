"""Audio processing and transcription stage.

Extracted from IngestionPipeline to reduce God class size.
IngestionPipeline inherits from AudioStageMixin to compose these methods.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import settings
from core.processing.text_utils import parse_srt
from core.processing.transcriber import AudioTranscriber
from core.storage.db import VectorDB
from core.utils.hardware import RESOURCE_ARBITER
from core.storage.constants import (
    MEDIA_SEGMENTS_COLLECTION
)

if TYPE_CHECKING:
    pass


class AudioStage:
    """Audio processing and transcription stage."""

    def __init__(self, db: VectorDB, get_probe_data, cleanup_memory, prepare_segments_for_db):
        self.db = db
        self.get_probe_data = get_probe_data
        self._cleanup_memory = cleanup_memory
        self.audio_classification: dict | None = None
        self._prepare_segments_for_db = prepare_segments_for_db

    async def process_audio(self, path: Path) -> None:
        """Processes audio to generate transcriptions and language classification.

        Prioritizes sidecar SRT files, then tries to extract embedded
        subtitles, and finally falls back to AI-based ASR (Whisper).
        Stores the resulting segments in the vector database.

        Args:
            path: Path to the media file.
        """
        from core.utils.logger import log

        audio_segments: list[dict[str, Any]] = []
        srt_path = path.with_suffix(".srt")

        # Check for existing sidecar SRT
        if srt_path.exists():
            audio_segments = parse_srt(srt_path) or []
            if audio_segments:
                log(
                    f"[Audio] Using existing SRT: {len(audio_segments)} segments"
                )

        # Check for embedded subtitles
        if not audio_segments:
            temp_srt = path.with_suffix(".embedded.srt")
            try:
                with AudioTranscriber() as transcriber:
                    if transcriber._find_existing_subtitles(
                        path, temp_srt, None, "ta"
                    ):
                        audio_segments = parse_srt(temp_srt) or []
                        if audio_segments:
                            log(
                                f"[Audio] Extracted embedded subs: {len(audio_segments)} segments"
                            )
            except Exception as e:
                log(f"[Audio] Embedded subtitle extraction failed: {e}")
            finally:
                if temp_srt.exists():
                    temp_srt.unlink()

        # Run ASR if no existing subtitles
        if not audio_segments:
            # Content Classification (speech/music/silence detection)
            use_lyrics_mode = False
            try:
                from core.processing.content_classifier import (
                    get_content_classifier,
                )

                classifier = get_content_classifier()
                content_regions = classifier.classify(path)

                if content_regions:
                    use_lyrics_mode = classifier.should_use_lyrics_mode(
                        content_regions
                    )
                    if use_lyrics_mode:
                        log(
                            "[Audio] High music content detected - will use lyrics mode"
                        )
            except Exception as e:
                log(f"[Audio] Content classification skipped: {e}")

            # Auto-detect language if enabled
            detected_lang = "en"
            detection_confidence = 0.0

            if settings.auto_detect_language:
                (
                    detected_lang,
                    detection_confidence,
                ) = await self._detect_audio_language_with_confidence(path)
                log(
                    f"[Audio] Detected language: {detected_lang} ({detection_confidence:.1%} confidence)"
                )

                # === DYNAMIC MULTI-PASS DETECTION ===
                # If confidence is low (<60%), try detecting on a different segment
                # This helps with music videos where intro might not have speech
                if detection_confidence < 0.6:
                    log(
                        f"[Audio] Low confidence ({detection_confidence:.1%}), trying second pass on different segment..."
                    )
                    (
                        second_lang,
                        second_conf,
                    ) = await self._detect_audio_language_with_confidence(
                        path, start_offset=30.0, duration=30.0
                    )
                    log(
                        f"[Audio] Second pass: {second_lang} ({second_conf:.1%})"
                    )

                    # Use the detection with higher confidence
                    if second_conf > detection_confidence:
                        detected_lang = second_lang
                        detection_confidence = second_conf
                        log(
                            f"[Audio] Using second pass result: {detected_lang}"
                        )

                    # If still low confidence, try third pass on middle of video
                    if detection_confidence < 0.5:
                        try:
                            # Probe for duration
                            probed = await self.get_probe_data(path)
                            duration = float(
                                probed.get("format", {}).get("duration", 0.0)
                            )
                            if (
                                duration > 120
                            ):  # Only if video is longer than 2 min
                                mid_point = duration / 2
                                (
                                    third_lang,
                                    third_conf,
                                ) = await self._detect_audio_language_with_confidence(
                                    path, start_offset=mid_point, duration=30.0
                                )
                                log(
                                    f"[Audio] Third pass (mid-video): {third_lang} ({third_conf:.1%})"
                                )
                                if third_conf > detection_confidence:
                                    detected_lang = third_lang
                                    detection_confidence = third_conf
                                    log(
                                        f"[Audio] Using mid-video detection: {detected_lang}"
                                    )
                        except Exception:
                            pass
            else:
                detected_lang = settings.language or "en"

            # Use Whisper for all languages (English, Indic, etc.)
            log(
                f"[Audio] Using Whisper turbo for '{detected_lang}'"
                + (" (lyrics mode)" if use_lyrics_mode else "")
            )
            try:
                with AudioTranscriber() as transcriber:
                    async with RESOURCE_ARBITER.acquire("whisper", vram_gb=1.5):
                        audio_segments = (
                            await transcriber.transcribe(
                                path,
                                language=detected_lang,
                                force_lyrics=use_lyrics_mode,
                            )
                            or []
                        )

                    # Auto-retry with lyrics mode if no segments and wasn't already lyrics mode
                    if not audio_segments and not use_lyrics_mode:
                        log(
                            "[Audio] No segments found, retrying with lyrics mode..."
                        )
                        audio_segments = (
                            await transcriber.transcribe(
                                path,
                                language=detected_lang,
                                force_lyrics=True,
                            )
                            or []
                        )
            except Exception as e:
                log(f"[Audio] Whisper failed: {e}")

            if audio_segments:
                log(
                    f"[Audio] Transcription SUCCESS: {len(audio_segments)} segments"
                )
            else:
                log(f"[Audio] WARNING - NO SEGMENTS produced for {path.name}")
                # NEVER-EMPTY GUARANTEE: Create a placeholder segment to preserve timeline
                # This ensures search can still find the media by path/timestamp
                try:
                    probed = await self.get_probe_data(path)
                    duration = float(
                        probed.get("format", {}).get("duration", 0.0)
                    )
                    if duration > 0:
                        # FIX #5: Empty text instead of "[No speech detected]"
                        # which was being embedded and polluting dialogue searches.
                        # Empty text → near-zero vector (Fix #1) → won't match real queries.
                        audio_segments = [
                            {
                                "text": "",
                                "start": 0.0,
                                "end": duration,
                                "_placeholder": True,
                            }
                        ]
                        log(
                            f"[Audio] Created placeholder segment for {duration:.1f}s media"
                        )
                except Exception:
                    pass

        if audio_segments:
            prepared = self._prepare_segments_for_db(
                path=path, chunks=audio_segments
            )
            await self.db.insert_media_segments(str(path), prepared)
            log(f"[Audio] Stored {len(prepared)} dialogue segments in DB")

            try:
                probed = await self.get_probe_data(path)
                total_duration = float(
                    probed.get("format", {}).get("duration", 0.0)
                )
                if total_duration > 0:
                    speech_duration = sum(
                        (s.get("end", 0) - s.get("start", 0))
                        for s in audio_segments
                        if s.get("text", "").strip()
                        and "[No speech" not in s.get("text", "")
                    )
                    speech_pct = (speech_duration / total_duration) * 100
                    music_pct = 100 - speech_pct
                    self.audio_classification = {
                        "music_percentage": music_pct,
                        "speech_percentage": speech_pct,
                        "total_duration": total_duration,
                    }
                    log(
                        f"[Audio] Classification: {speech_pct:.0f}% speech, {music_pct:.0f}% music/ambience"
                    )
            except Exception:
                pass

        # NOTE: CLAP detection removed — redundant with _process_audio_events()
        # which uses AST dynamic prediction (527 AudioSet classes) + CLAP embeddings.
        # See _process_audio_events() for the definitive audio event detection path.

        # ============================================================
        # AUDIO LOUDNESS ANALYSIS: Detect loud moments (e.g., "92dB cheer")
        # Uses pyloudnorm for ITU-R BS.1770-4 compliant loudness measurement
        # ============================================================
        try:
            log(
                "[Loudness] Starting audio level analysis (FFmpeg streaming)..."
            )

            # === FFmpeg EBUR128 LOUDNESS ANALYSIS (OOM-SAFE) ===
            # Uses FFmpeg's ebur128 filter which streams the audio (~50MB RAM max)
            # instead of loading entire file into RAM (3-4GB for long videos)
            import re
            import subprocess

            try:
                # Run FFmpeg with ebur128 loudness filter
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(path),
                    "-af",
                    "ebur128=framelog=verbose:peak=true",
                    "-f",
                    "null",
                    "-",
                ]
                result = await asyncio.to_thread(
                    lambda: subprocess.run(
                        cmd, capture_output=True, text=True, timeout=300
                    )
                )

                # Parse the summary line from stderr
                # Format: "Summary: Integrated loudness: -23.0 LUFS, Loudness range: 5.0 LU"
                stderr = result.stderr

                lufs = -23.0  # Default
                peak_db = 0.0

                # Extract integrated loudness
                lufs_match = re.search(r"I:\s*(-?\d+\.?\d*)\s*LUFS", stderr)
                if lufs_match:
                    lufs = float(lufs_match.group(1))

                # Extract true peak
                peak_match = re.search(r"Peak:\s*(-?\d+\.?\d*)\s*dBFS", stderr)
                if peak_match:
                    peak_db = float(peak_match.group(1))

                # Estimate SPL from LUFS (rough conversion)
                estimated_spl = max(
                    0, 85 + lufs + 23
                )  # 85dB baseline at -23 LUFS

                # Categorize
                if estimated_spl < 60:
                    category = "quiet"
                elif estimated_spl < 75:
                    category = "moderate"
                elif estimated_spl < 85:
                    category = "loud"
                else:
                    category = "very_loud"

                log(
                    f"[Loudness] Overall: {estimated_spl:.0f} dB SPL ({category}) [LUFS: {lufs:.1f}]"
                )

                # Store overall loudness in media metadata
                self.db.update_media_metadata(
                    media_path=str(path),
                    metadata={
                        "loudness_lufs": lufs,
                        "peak_db": peak_db,
                        "estimated_spl": estimated_spl,
                        "loudness_category": category,
                    },
                )
            except Exception as e:
                log(f"[Loudness] FFmpeg analysis failed: {e}")
        except Exception as e:
            log(f"[Loudness] Analysis failed: {e}")

        # ============================================================
        # MUSIC STRUCTURE ANALYSIS: Detect verse/chorus/bridge/drop
        # Enables temporal queries like "during the chorus" or "at the drop"
        # ============================================================
        try:
            if getattr(settings, "enable_music_structure", False):
                from core.processing.audio_structure import get_music_analyzer
                import asyncio
    
                log("[MusicStructure] Starting structure analysis...")
                music_analyzer = get_music_analyzer()
    
                # Load audio if not already loaded
                # SAFETY: Only load first 5 minutes for music structure to prevent OOM
                # Music structure (verse/chorus) is typically established early
                import librosa
    
                max_duration = 300.0  # 5 minutes max for music analysis
                audio_array, sr = await asyncio.to_thread(
                    librosa.load, str(path), sr=22050, mono=True, duration=max_duration
                )
                log(
                    f"[MusicStructure] Loaded {len(audio_array) / sr:.1f}s audio (limited to {max_duration}s)"
                )
    
                # Analyze music structure
                analysis = await asyncio.to_thread(
                    music_analyzer.analyze_array, audio_array, sr=22050
                )

                if analysis.sections:
                    log(
                        f"[MusicStructure] Found {len(analysis.sections)} sections at {analysis.global_tempo:.1f} BPM"
                    )

                    # Store each section as an audio event for searchability
                    for section in analysis.sections:
                        self.db.insert_audio_event(
                            media_path=str(path),
                            event_type=f"music_{section.section_type}",
                            start_time=section.start_time,
                            end_time=section.end_time,
                            confidence=section.confidence,
                            payload={
                                "section_type": section.section_type,
                                "energy": section.energy,
                                "beat_count": section.beat_count,
                                "tempo": section.tempo,
                            },
                        )

                    # Store music metadata
                    self.db.update_media_metadata(
                        media_path=str(path),
                        metadata={
                            "music_tempo": analysis.global_tempo,
                            "has_vocals": analysis.has_vocals,
                            "section_count": len(analysis.sections),
                            "music_structure": [
                                s.to_dict() for s in analysis.sections[:20]
                            ],  # Limit for storage
                        },
                    )
                    log(
                        f"[MusicStructure] Indexed {len(analysis.sections)} sections"
                    )
                else:
                    log("[MusicStructure] No sections detected (may not be music)")

        except Exception as e:
            log(f"[MusicStructure] Analysis failed: {e}")

        self._cleanup_memory()

    async def _detect_audio_language(self, path: Path) -> str:
        """Detect audio language. Delegates to language_detection module."""
        from core.ingestion.language_detection import detect_audio_language

        return await detect_audio_language(path)

    async def _detect_audio_language_with_confidence(
        self,
        path: Path,
        start_offset: float = 0.0,
        duration: float = 30.0,
    ) -> tuple[str, float]:
        """Detect audio language with confidence. Delegates to language_detection module."""
        from core.ingestion.language_detection import (
            detect_audio_language_with_confidence,
        )

        return await detect_audio_language_with_confidence(
            path, start_offset=start_offset, duration=duration
        )
