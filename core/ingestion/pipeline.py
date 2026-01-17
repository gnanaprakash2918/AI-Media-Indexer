"""Media ingestion pipeline orchestrator."""

from __future__ import annotations

import asyncio
import gc
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from qdrant_client.http import models

from config import settings
from core.llm.vlm_factory import get_vlm_client
from core.processing.extractor import FrameExtractor
from core.processing.frame_sampling import TextGatedOCR
from core.processing.identity import FaceManager, FaceTrackBuilder
from core.processing.metadata import MetadataEngine
from core.processing.ocr import EasyOCRProcessor
from core.processing.prober import MediaProbeError, MediaProber
from core.processing.scene_detector import detect_scenes, extract_scene_frame
from core.processing.segmentation import Sam3Tracker
from core.processing.temporal_context import (
    SceneletBuilder,
    TemporalContextManager,
)
from core.processing.text_utils import parse_srt
from core.processing.transcriber import AudioTranscriber
from core.processing.vision import VisionAnalyzer
from core.processing.voice import VoiceProcessor
from core.schemas import MediaType
from core.storage.db import VectorDB
from core.storage.identity_graph import identity_graph
from core.utils.frame_sampling import FrameSampler
from core.utils.logger import bind_context, logger
from core.utils.observe import observe
from core.utils.progress import progress_tracker
from core.utils.resource import resource_manager
from core.utils.resource_arbiter import GPU_SEMAPHORE, RESOURCE_ARBITER
from core.utils.retry import retry


class IngestionPipeline:
    """Orchestrate the media ingestion process (probing, transcription, vision, etc)."""

    def __init__(
        self,
        *,
        qdrant_backend: str = settings.qdrant_backend,
        qdrant_host: str = settings.qdrant_host,
        qdrant_port: int = settings.qdrant_port,
        frame_interval_seconds: float = settings.frame_interval,
        tmdb_api_key: str | None = settings.tmdb_api_key,
    ) -> None:
        """Initializes the ingestion pipeline and its sub-components.

        Args:
            qdrant_backend: The storage backend ('memory' or 'docker').
            qdrant_host: Qdrant host address for docker backend.
            qdrant_port: Qdrant port for docker backend.
            frame_interval_seconds: Interval between sampled frames in seconds.
            tmdb_api_key: Optional API key for TMDB movie metadata.
        """
        self.scene_detector = detect_scenes
        self.face_manager = FaceManager()
        self.vision_analyzer = VisionAnalyzer()
        self.audio_transcriber = AudioTranscriber()
        self.prober = MediaProber()
        self.sam_tracker = Sam3Tracker()
        self.metadata_engine = MetadataEngine(
            tmdb_api_key=settings.tmdb_api_key
        )
        self.voice_processor = VoiceProcessor()

        # OCR Components
        self.ocr_engine = EasyOCRProcessor(langs=["en"], use_gpu=True)
        self.text_gate = TextGatedOCR()
        self.extractor = FrameExtractor()
        self.db = VectorDB(
            backend=qdrant_backend,
            host=qdrant_host,
            port=qdrant_port,
        )
        self.vision_analyzer = VisionAnalyzer()
        self.metadata_engine = MetadataEngine(
            tmdb_key=settings.tmdb_api_key, omdb_key=settings.omdb_api_key
        )
        self.face_manager = FaceManager(
            db_client=self.db.client,
            dbscan_eps=settings.hdbscan_cluster_selection_epsilon,
            dbscan_min_samples=settings.hdbscan_min_samples,
        )
        self.voice_processor = VoiceProcessor(db=self.db)
        self.frame_interval_seconds = frame_interval_seconds
        self.vision: VisionAnalyzer | None = None
        self.faces: FaceManager | None = None
        self.voice: VoiceProcessor | None = None

        # Enhanced pipeline config for SmartFrameSampler, BiometricArbitrator, etc.
        self._enhanced_config = None
        self._face_clusters: dict[int, list[float]] = {}

        # Deep Video Understanding (SAM 3)
        self.sam3_tracker = (
            Sam3Tracker() if settings.enable_sam3_tracking else None
        )
        self.frame_sampler = FrameSampler(every_n=5)

    @property
    def enhanced_config(self):
        """Lazy-load EnhancedPipelineConfig for SmartFrameSampler, audio events, etc."""
        if self._enhanced_config is None:
            try:
                from core.integration import get_enhanced_config

                self._enhanced_config = get_enhanced_config()
                logger.info("[Pipeline] EnhancedPipelineConfig loaded")
            except Exception as e:
                logger.warning(f"[Pipeline] EnhancedPipelineConfig failed: {e}")
        return self._enhanced_config

    def _cleanup_memory(self, context: str = "") -> None:
        """Force garbage collection and clear CUDA cache.

        Args:
            context: Optional context string for logging (e.g., "audio_complete")
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all GPU operations complete

        # Log VRAM status if available
        try:
            from core.utils.hardware import log_vram_status

            log_vram_status(context or "cleanup")
        except Exception:
            pass

    @observe("process_video")
    async def process_video(
        self,
        video_path: str | Path,
        media_type_hint: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        job_id: str | None = None,
        content_type_hint: str = "auto",
    ) -> str:
        """Orchestrates the full ingestion of a video file.

        Executes audio transcription, voice diarization, visual frame analysis,
        and scene-level summarization. Supports resume from crash if job_id
        is provided.

        Args:
            video_path: Path to the media file to ingest.
            media_type_hint: Optional hint ('movie', 'episode', etc).
            start_time: Optional start timestamp for clipped ingestion.
            end_time: Optional end timestamp for clipped ingestion.
            job_id: Optional existing job ID for resuming after a crash.
            content_type_hint: Optional hint for processing ('song', etc).

        Returns:
            The job ID associated with this ingestion task.
        """
        resume = False
        if job_id:
            resume = True
        else:
            job_id = str(uuid.uuid4())

        bind_context(component="pipeline")

        self._start_time = start_time
        self._end_time = end_time
        self._hitl_content_type = (
            content_type_hint if content_type_hint != "auto" else None
        )
        self._audio_classification = None

        path = Path(video_path)
        progress_tracker.start(
            job_id,
            file_path=str(path),
            media_type=media_type_hint or "unknown",
            resume=resume,
        )

        if not path.exists() or not path.is_file():
            progress_tracker.fail(job_id, error=f"Invalid media path: {path}")
            raise FileNotFoundError(f"Invalid media path: {path}")

        hint_enum = (
            MediaType(media_type_hint)
            if media_type_hint in MediaType._value2member_map_
            else MediaType.UNKNOWN
        )

        _ = await self.metadata_engine.identify(path, user_hint=hint_enum)

        try:
            probed = self.prober.probe(path)
            duration = float(probed.get("format", {}).get("duration", 0.0))
        except MediaProbeError as e:
            progress_tracker.fail(job_id, error=f"Media probe failed: {e}")
            raise

        # RESUME LOGIC: Check checkpoint for crash recovery
        checkpoint = None
        skip_audio = False
        skip_voice = False
        resume_from_frame = 0
        if resume:
            from core.ingestion.jobs import job_manager

            existing_job = job_manager.get_job(job_id)
            if existing_job and existing_job.checkpoint_data:
                checkpoint = existing_job.checkpoint_data
                skip_audio = checkpoint.get("audio_complete", False)
                skip_voice = checkpoint.get("voice_complete", False)
                resume_from_frame = checkpoint.get("last_frame", 0)
                logger.info(
                    f"Resuming job {job_id}: skip_audio={skip_audio}, skip_voice={skip_voice}, resume_from={resume_from_frame}"
                )

        self._resume_from_frame = resume_from_frame  # Store for _process_frames

        try:
            if not skip_audio:
                progress_tracker.update(
                    job_id, 5.0, stage="audio", message="Processing audio"
                )
                await retry(lambda: self._process_audio(path))
                logger.info("[Pipeline] _process_audio completed, running cleanup...")
                self._cleanup_memory("audio_complete")  # Unload Whisper
                logger.info("[Pipeline] Audio cleanup done, saving checkpoint...")
                # Checkpoint audio completion
                progress_tracker.save_checkpoint(
                    job_id, {"audio_complete": True}
                )
            logger.info("[Pipeline] Audio phase complete, moving to voice processing...")
            progress_tracker.update(
                job_id, 30.0, stage="audio", message="Audio complete"
            )

            if progress_tracker.is_cancelled(job_id):
                return job_id
            if progress_tracker.is_paused(job_id):
                return job_id

            if not skip_voice:
                progress_tracker.update(
                    job_id, 35.0, stage="voice", message="Processing voice"
                )
                await retry(lambda: self._process_voice(path))
                self._cleanup_memory("voice_complete")  # Unload Pyannote
                # Checkpoint voice completion
                progress_tracker.save_checkpoint(
                    job_id, {"voice_complete": True}
                )
            progress_tracker.update(
                job_id, 50.0, stage="voice", message="Voice complete"
            )

            logger.debug("Voice complete - checking job status")

            if progress_tracker.is_cancelled(job_id):
                logger.info(f"Job {job_id} cancelled")
                return job_id

            if progress_tracker.is_paused(job_id):
                logger.info(f"Job {job_id} paused")
                return job_id

            progress_tracker.update(
                job_id, 55.0, stage="frames", message="Processing frames"
            )
            logger.debug("Starting frame processing")
            await retry(
                lambda: self._process_frames(
                    path, job_id, total_duration=duration
                )
            )

            logger.debug("Frame processing complete - cleaning memory")
            self._cleanup_memory("frames_complete")

            if progress_tracker.is_cancelled(
                job_id
            ) or progress_tracker.is_paused(job_id):
                return job_id

            # Dense Scene Captioning (VLM on detected scene boundaries)
            progress_tracker.update(
                job_id,
                90.0,
                stage="scene_captions",
                message="Generating scene captions",
            )
            await self._process_scene_captions(path, job_id)

            # Post-Processing Phase
            progress_tracker.update(
                job_id,
                95.0,
                stage="post_processing",
                message="Enriching metadata",
            )
            await self._post_process_video(path, job_id)

            progress_tracker.complete(job_id, message=f"Completed: {path.name}")

            return job_id

        except Exception as e:
            progress_tracker.fail(job_id, error=str(e))
            raise

    @observe("audio_processing")
    async def _process_audio(self, path: Path) -> None:
        """Processes audio to generate transcriptions and language classification.

        Prioritizes sidecar SRT files, then tries to extract embedded
        subtitles, and finally falls back to AI-based ASR (Whisper or AI4Bharat).
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
            await resource_manager.throttle_if_needed("compute")
            try:
                with AudioTranscriber() as transcriber:
                    temp_srt = path.with_suffix(".embedded.srt")
                    if transcriber._find_existing_subtitles(
                        path, temp_srt, None, "ta"
                    ):
                        audio_segments = parse_srt(temp_srt) or []
                        if audio_segments:
                            log(
                                f"[Audio] Extracted embedded subs: {len(audio_segments)} segments"
                            )
                        if temp_srt.exists():
                            temp_srt.unlink()
            except Exception as e:
                log(f"[Audio] Embedded subtitle extraction failed: {e}")

        # Run ASR if no existing subtitles
        if not audio_segments:
            await resource_manager.throttle_if_needed("compute")

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
            if settings.auto_detect_language:
                detected_lang = await self._detect_audio_language(path)
                log(f"[Audio] Detected language: {detected_lang}")
            else:
                detected_lang = settings.language or "en"

            # Choose transcriber based on language
            indic_languages = [
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

            if settings.use_indic_asr and detected_lang in indic_languages:
                # Use AI4Bharat for Indic languages
                log(
                    f"[Audio] Attempting AI4Bharat IndicConformer for '{detected_lang}'"
                )
                indic_transcriber = None
                try:
                    from core.processing.indic_transcriber import (
                        IndicASRPipeline,
                    )

                    indic_transcriber = IndicASRPipeline(lang=detected_lang)
                    log(
                        f"[Audio] IndicASR backend: {indic_transcriber._backend}"
                    )

                    # Generate SRT sidecar file alongside the video
                    srt_path = path.with_suffix(".srt")
                    async with GPU_SEMAPHORE:
                        audio_segments = (
                            indic_transcriber.transcribe(
                                path, output_srt=srt_path
                            )
                            or []
                        )

                    if audio_segments:
                        log(
                            f"[Audio] AI4Bharat SUCCESS: {len(audio_segments)} segments"
                        )
                        log(f"[Audio] SRT saved to: {srt_path}")
                    else:
                        log(
                            "[Audio] AI4Bharat returned empty, falling back to Whisper"
                        )
                        raise ValueError("AI4Bharat returned no segments")
                except Exception as e:
                    log(f"[Audio] AI4Bharat failed: {e}")
                    log(
                        f"[Audio] Falling back to Whisper for '{detected_lang}'"
                    )
                    try:
                        with AudioTranscriber() as transcriber:
                            # Use Arbiter to guarantee VRAM availability
                            async with RESOURCE_ARBITER.acquire(
                                "whisper", vram_gb=1.5
                            ):
                                audio_segments = (
                                    transcriber.transcribe(
                                        path, language=detected_lang
                                    )
                                    or []
                                )
                    except Exception as e2:
                        log(f"[Audio] Whisper fallback also failed: {e2}")
                finally:
                    # CRITICAL: Unload IndicASR to free VRAM for Ollama vision
                    if indic_transcriber is not None:
                        indic_transcriber.unload_model()
            else:
                # Use Whisper for English and other languages
                log(
                    f"[Audio] Using Whisper turbo for '{detected_lang}'"
                    + (" (lyrics mode)" if use_lyrics_mode else "")
                )
                try:
                    with AudioTranscriber() as transcriber:
                        async with RESOURCE_ARBITER.acquire(
                            "whisper", vram_gb=1.5
                        ):
                            audio_segments = (
                                transcriber.transcribe(
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
                                    transcriber.transcribe(
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
                    probed = self.prober.probe(path)
                    duration = float(
                        probed.get("format", {}).get("duration", 0.0)
                    )
                    if duration > 0:
                        audio_segments = [
                            {
                                "text": "[No speech detected]",
                                "start": 0.0,
                                "end": duration,
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
            self.db.insert_media_segments(str(path), prepared)
            log(f"[Audio] Stored {len(prepared)} dialogue segments in DB")

            try:
                probed = self.prober.probe(path)
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
                    self._audio_classification = {
                        "speech_percentage": min(speech_pct, 100),
                        "music_percentage": max(music_pct, 0),
                        "total_duration": total_duration,
                    }
                    log(
                        f"[Audio] Classification: {speech_pct:.0f}% speech, {music_pct:.0f}% music/ambience"
                    )
            except Exception:
                pass

        # ============================================================
        # CLAP AUDIO EVENT DETECTION: Previously dead code - now wired!
        # Detects non-speech sounds: bells, cheers, sirens, applause, etc.
        # ============================================================
        try:
            if self.enhanced_config and getattr(
                self.enhanced_config, "enable_clap", False
            ):
                import librosa

                from core.processing.audio_events import AudioEventDetector

                log("[CLAP] Starting audio event detection...")
                audio_detector = AudioEventDetector()

                # Load audio for CLAP (16kHz mono)
                try:
                    log("[CLAP] Loading audio file with librosa...")
                    # Run librosa.load in a thread to verify if it blocks
                    audio_array, sr = await asyncio.to_thread(
                        librosa.load, str(path), sr=16000, mono=True
                    )
                    log(f"[CLAP] Audio loaded: {len(audio_array)} samples, {sr}Hz")

                    # Process in 2-second chunks to detect events with timestamps
                    chunk_duration = 2.0
                    chunk_samples = int(chunk_duration * sr)
                    audio_events = []

                    for i in range(0, len(audio_array), chunk_samples):
                        chunk = audio_array[i : i + chunk_samples]
                        if len(chunk) < sr:  # Skip chunks < 1 second
                            continue

                        start_time = i / sr
                        events = await audio_detector.detect_events(
                            chunk,
                            sample_rate=int(sr),
                            top_k=3,
                            threshold=0.25,
                            target_classes=[
                                "applause",
                                "cheering",
                                "laughter",
                                "crowd",
                                "music",
                                "singing",
                                "scary music",
                                "happy music",
                                "siren",
                                "explosion",
                                "gunshot",
                                "glass breaking",
                                "dog barking",
                                "cat meowing",
                                "bird chirping",
                                "thunder",
                                "rain",
                                "ocean waves",
                                "wind",
                                "footsteps",
                                "door slamming",
                                "car engine",
                            ],
                        )

                        for event in events:
                            if event.get("event") not in ("speech", "silence"):
                                audio_events.append(
                                    {
                                        "event": event["event"],
                                        "confidence": event["confidence"],
                                        "start": start_time,
                                        "end": start_time + chunk_duration,
                                    }
                                )

                    if audio_events:
                        log(
                            f"[CLAP] Detected {len(audio_events)} audio events: {[e['event'] for e in audio_events[:5]]}"
                        )
                        # Store audio events in database
                        for event in audio_events:
                            self.db.insert_audio_event(
                                media_path=str(path),
                                event_type=event["event"],
                                start_time=event["start"],
                                end_time=event["end"],
                                confidence=event["confidence"],
                            )
                    else:
                        log("[CLAP] No non-speech audio events detected")

                except Exception as e:
                    log(f"[CLAP] Audio loading failed: {e}")
        except Exception as e:
            log(f"[CLAP] Audio event detection failed: {e}")

        # ============================================================
        # AUDIO LOUDNESS ANALYSIS: Detect loud moments (e.g., "92dB cheer")
        # Uses pyloudnorm for ITU-R BS.1770-4 compliant loudness measurement
        # ============================================================
        try:
            from core.processing.audio_levels import AudioLoudnessAnalyzer

            log("[Loudness] Starting audio level analysis...")
            loudness_analyzer = AudioLoudnessAnalyzer()

            # Load audio if not already loaded
            if "audio_array" not in locals():
                import librosa

                audio_array, sr = librosa.load(str(path), sr=16000, mono=True)

            # Analyze overall loudness
            loudness_result = await loudness_analyzer.analyze(audio_array)
            log(
                f"[Loudness] Overall: {loudness_result['estimated_spl']} dB SPL ({loudness_result['loudness_category']})"
            )

            # Detect loud moments (useful for queries like "crowd cheer at 92dB")
            loud_moments = await loudness_analyzer.detect_loud_moments(
                audio_array,
                threshold_spl=80,  # Detect moments above 80 dB
            )

            if loud_moments:
                log(f"[Loudness] Found {len(loud_moments)} loud moments")
                # Store loud moments in database
                for moment in loud_moments:
                    self.db.insert_audio_event(
                        media_path=str(path),
                        event_type=f"loud_moment_{moment['category']}",
                        start_time=moment["start"],
                        end_time=moment["end"],
                        confidence=0.9,  # High confidence for dB-based detection
                        payload={"peak_spl": moment["peak_spl"]},
                    )

            # Store overall loudness in media metadata
            self.db.update_media_metadata(
                media_path=str(path),
                metadata={
                    "loudness_lufs": loudness_result["lufs"],
                    "peak_db": loudness_result["peak_db"],
                    "estimated_spl": loudness_result["estimated_spl"],
                    "loudness_category": loudness_result["loudness_category"],
                },
            )
        except Exception as e:
            log(f"[Loudness] Analysis failed: {e}")

        self._cleanup_memory()

    async def _detect_audio_language(self, path: Path) -> str:
        """Detects the audio language using Whisper's language detection.

        Args:
            path: Path to the media file.

        Returns:
            The detected ISO 639-1 language code (e.g., 'en', 'ta', 'hi').
        """
        await resource_manager.throttle_if_needed("compute")

        # Run detection in a thread to not block asyncio loop
        # (Whisper is blocking)
        try:
            return await asyncio.to_thread(self._run_detection, path)
        except Exception as e:
            from core.utils.logger import log

            log(f"[Audio] Language detection failed: {e}")
            return "en"

    def _run_detection(self, path: Path) -> str:
        """Synchronous helper for language detection.

        Args:
            path: Path to the media file.

        Returns:
            The detected language code.
        """
        with AudioTranscriber() as transcriber:
            return transcriber.detect_language(path)

    @observe("voice_processing")
    async def _process_voice(self, path: Path) -> None:
        """Processes voice diarization and identity registries.

        Extracts voice segments, generates embeddings, matches them against
        the global speaker registry, and stores them in the database. Also
        extracts audio clips for each identified voice segment.

        Args:
            path: Path to the media file.
        """
        await resource_manager.throttle_if_needed("compute")
        self.voice = VoiceProcessor()

        try:
            voice_segments = await self.voice.process(path)

            # Prepare voice thumbnails directory
            thumb_dir = settings.cache_dir / "thumbnails" / "voices"
            thumb_dir.mkdir(parents=True, exist_ok=True)

            import hashlib
            import subprocess

            # Create safe prefix
            safe_stem = hashlib.md5(path.stem.encode()).hexdigest()

            # Global Speaker Registry Logic
            # 1. Match against existing speakers
            # 2. Assign Global ID
            # 3. Persist specific samples for future matching

            import random  # For cluster ID generation

            for _idx, seg in enumerate(voice_segments or []):
                audio_path: str | None = None
                global_speaker_id = f"unknown_{uuid.uuid4().hex[:8]}"
                voice_cluster_id = -1

                # Check Global Registry if embedding exists
                # Check Global Registry if embedding exists
                if seg.embedding is not None:
                    match = self.db.match_speaker(
                        seg.embedding,
                        threshold=settings.voice_clustering_threshold,
                    )
                    if match:
                        global_speaker_id, existing_cluster_id, _score = match
                        voice_cluster_id = existing_cluster_id
                        # If existing matched speaker has no cluster ID (-1), generate one?
                        # Usually it should have one if we follow this new logic consistently.
                        if voice_cluster_id == -1:
                            voice_cluster_id = random.randint(100000, 999999)
                    else:
                        # New Global Speaker -> New Cluster
                        global_speaker_id = f"SPK_{uuid.uuid4().hex[:12]}"
                        voice_cluster_id = random.randint(100000, 999999)

                        self.db.upsert_speaker_embedding(
                            speaker_id=global_speaker_id,
                            embedding=seg.embedding,
                            media_path=str(path),
                            start=seg.start_time,
                            end=seg.end_time,
                            voice_cluster_id=voice_cluster_id,
                        )

                # ALWAYS extract audio clip for every segment (not just those with embeddings)
                try:
                    clip_name = f"{safe_stem}_{seg.start_time:.2f}_{seg.end_time:.2f}.mp3"
                    clip_file = thumb_dir / clip_name

                    if not clip_file.exists():
                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(path),
                            "-ss",
                            str(seg.start_time),
                            "-to",
                            str(seg.end_time),
                            "-q:a",
                            "2",  # High quality MP3
                            "-map",
                            "a",
                            "-loglevel",
                            "error",
                            str(clip_file),
                        ]
                        subprocess.run(
                            cmd, capture_output=True, text=True
                        )
                except Exception as e:
                    logger.warning(
                        f"[Voice] FFmpeg failed for {clip_name}: {e}"
                    )

                if clip_file.exists():
                    audio_path = f"/thumbnails/voices/{clip_name}"
                else:
                    logger.warning(
                        f"[Voice] Audio clip missing after ffmpeg: {clip_file}"
                    )

                # ALWAYS store voice segment (even if no embedding, use placeholder)
                # Only store if we have an embedding (required by insert_voice_segment)
                if seg.embedding is not None and audio_path:
                    self.db.insert_voice_segment(
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        speaker_label=global_speaker_id,  # Use Global ID
                        embedding=seg.embedding,
                        audio_path=audio_path,
                        voice_cluster_id=voice_cluster_id,
                    )
                else:
                    # Log segments without embeddings for debugging
                    logger.warning(
                        f"[Voice] Segment {seg.start_time:.2f}-{seg.end_time:.2f}s has no embedding, audio_path={audio_path}"
                    )
        finally:
            if self.voice:
                self.voice.cleanup()
            del self.voice
            self.voice = None
            self._cleanup_memory()

    @observe("frame_processing")
    async def _process_frames(
        self, path: Path, job_id: str | None = None, total_duration: float = 0.0
    ) -> None:
        """Handles visual frame extraction and vision analysis.

        Samples frames at a fixed interval, performs face detection and
        VLM analysis for each sampled frame, and builds temporal face tracks.
        Supports resume via checkpointing and manages memory/throttling.

        Args:
            path: Path to the media file.
            job_id: Optional ID for progress tracking and checkpointing.
            total_duration: Total video duration for accurate progress reporting.
        """
        vision_task_type = (
            "network" if settings.llm_provider == "gemini" else "compute"
        )
        await resource_manager.throttle_if_needed(vision_task_type)

        # Use the configured LLM provider from settings
        from llm.factory import LLMFactory

        vision_llm = LLMFactory.create_llm(provider=settings.llm_provider.value)
        self.vision = VisionAnalyzer(llm=vision_llm)

        # GLOBAL IDENTITY: Load existing cluster centroids from DB
        # This enables cross-video identity matching (O(1) gallery-probe)
        global_clusters = self.db.get_all_cluster_centroids()
        logger.info(
            f"[GlobalIdentity] Loaded {len(global_clusters)} cluster centroids for matching"
        )

        # InsightFace uses GPU for fast detection, but we unload it before Ollama
        self.faces = FaceManager(
            dbscan_eps=0.55,  # HDBSCAN cluster_selection_epsilon
            dbscan_min_samples=3,  # HDBSCAN min_samples
            use_gpu=settings.device == "cuda",
            global_clusters=global_clusters,
        )

        # Initialize FaceTrackBuilder for temporal face grouping
        # This creates stable per-video tracks before global identity linking
        self._face_track_builder = FaceTrackBuilder(
            frame_interval=float(self.frame_interval_seconds)
        )
        # Store video path for Identity Graph
        self._current_media_id = str(path)

        # Pass time range to extractor for partial processing
        frame_generator = self.extractor.extract(
            path,
            interval=self.frame_interval_seconds,
            start_time=getattr(self, "_start_time", None),
            end_time=getattr(self, "_end_time", None),
        )

        # Time offset for accurate timestamps in partial extraction
        time_offset = getattr(self, "_start_time", None) or 0.0

        frame_count = 0

        # Resume support: skip already processed frames
        resume_from_frame = getattr(self, "_resume_from_frame", 0)

        # XMem-style temporal context for video coherence
        from core.processing.temporal_context import (
            TemporalContext,
        )

        temporal_ctx = TemporalContextManager(sensory_size=5)

        # Scenelet Builder (Sliding Window: 5s window, 2.5s stride)
        scenelet_builder = SceneletBuilder(
            window_seconds=5.0, stride_seconds=2.5
        )
        scenelet_builder.set_audio_segments(
            self._get_audio_segments_for_video(str(path))
        )

        async for frame_path in frame_generator:
            if job_id:
                if progress_tracker.is_cancelled(
                    job_id
                ) or progress_tracker.is_paused(job_id):
                    break

            # RESUME: Skip already processed frames
            if frame_count < resume_from_frame:
                frame_count += 1
                if frame_path.exists():
                    frame_path.unlink()
                continue

            # Calculate actual timestamp including offset
            timestamp = time_offset + (
                frame_count * float(self.frame_interval_seconds)
            )
            if self.frame_sampler.should_sample(frame_count):
                # Get temporal context from XMem-style memory
                narrative_context = temporal_ctx.get_context_for_vlm()
                neighbor_timestamps = [
                    c.timestamp for c in temporal_ctx.sensory
                ]

                new_desc = await self._process_single_frame(
                    video_path=path,
                    frame_path=frame_path,
                    timestamp=timestamp,
                    index=frame_count,
                    context=narrative_context,
                    neighbor_timestamps=neighbor_timestamps,
                )
                if new_desc:
                    # Add to temporal context memory
                    t_ctx = TemporalContext(
                        timestamp=timestamp,
                        description=new_desc[:200],
                        faces=list(self._face_clusters.keys())
                        if hasattr(self, "_face_clusters")
                        else [],
                    )
                    temporal_ctx.add_frame(t_ctx)

                    # Add to Scenelet Builder
                    # Use full description for scenelets
                    s_ctx = TemporalContext(
                        timestamp=timestamp,
                        description=new_desc,
                        faces=list(self._face_clusters.keys())
                        if hasattr(self, "_face_clusters")
                        else [],
                    )
                    scenelet_builder.add_frame(s_ctx)

            if job_id:
                if progress_tracker.is_paused(job_id):
                    logger.info(f"Job {job_id} paused. Stopping frame loop.")
                    break

                if progress_tracker.is_cancelled(job_id):
                    logger.warning(
                        f"Job {job_id} cancelled. Aborting pipeline."
                    )
                    return

                # Update Granular Stats
                # Use provided duration for 100% accuracy, fallback to metadata
                video_duration = total_duration
                if not video_duration:
                    try:
                        probe_data = self.prober.probe(path)
                        video_duration = float(
                            probe_data.get("format", {}).get("duration", 0.0)
                        )
                    except Exception:
                        video_duration = 0.0

                interval = float(self.frame_interval_seconds)

                total_est_frames = (
                    int(video_duration / interval) if video_duration else 0
                )
                current_ts = timestamp
                current_frame_index = (
                    int(current_ts / interval) if interval > 0 else frame_count
                )

                status_msg = f"Processing frame {current_frame_index}/{total_est_frames} at {current_ts:.1f}s"

                progress_tracker.update_granular(
                    job_id,
                    processed_frames=current_frame_index,
                    total_frames=total_est_frames,
                    current_timestamp=current_ts,
                    total_duration=video_duration,
                )

                if frame_count % 5 == 0:
                    progress = (
                        55.0
                        + min(
                            40.0,
                            (current_ts / (video_duration or 1)) * 40.0,
                        )
                        if video_duration
                        else 55.0
                    )
                    progress_tracker.update(
                        job_id,
                        progress,
                        stage="frames",
                        message=status_msg,
                    )

            await asyncio.sleep(0.01)  # Minimal sleep for responsiveness
            # Always delete the frame file after processing
            if frame_path.exists():
                try:
                    frame_path.unlink()
                except Exception:
                    pass

            frame_count += 1

            # Aggressive memory cleanup every 5 frames to prevent OOM
            # This preserves timestamps and accuracy while managing VRAM
            cleanup_interval = 5
            if frame_count % cleanup_interval == 0:
                self._cleanup_memory(context=f"frame_{frame_count}")
                # Extra VRAM flush for video processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Thermal throttling - pause if system overheating
                await resource_manager.throttle_if_needed("compute")

            # CHECKPOINT: Save progress every 50 frames for crash recovery
            checkpoint_interval = 50
            if job_id and frame_count % checkpoint_interval == 0:
                from core.ingestion.jobs import job_manager

                checkpoint_data = {
                    "last_frame": frame_count,
                    "last_timestamp": timestamp,
                    "audio_complete": True,
                    "voice_complete": True,
                    "frames_complete": False,
                }
                job_manager.update_job(
                    job_id,
                    checkpoint_data=checkpoint_data,
                    processed_frames=frame_count,
                    current_frame_timestamp=timestamp,
                )
                job_manager.update_heartbeat(job_id)
                logger.debug(f"Checkpoint saved at frame {frame_count}")

        # Finalize face tracks and store in Identity Graph
        # This is the key step: convert frame-by-frame detections into stable tracks
        if hasattr(self, "_face_track_builder") and self._face_track_builder:
            try:
                finalized_tracks = self._face_track_builder.finalize_all()
                logger.info(
                    f"Finalized {len(finalized_tracks)} face tracks for {path.name}"
                )

                # Store each track in the Identity Graph
                media_id = getattr(self, "_current_media_id", str(path))
                for (
                    _track_id,
                    avg_embedding,
                    metadata,
                ) in self._face_track_builder.get_track_embeddings():
                    try:
                        identity_graph.create_face_track(
                            media_id=media_id,
                            start_frame=metadata["start_frame"],
                            end_frame=metadata["end_frame"],
                            start_time=metadata["start_time"],
                            end_time=metadata["end_time"],
                            avg_embedding=avg_embedding,
                            avg_confidence=metadata.get("avg_confidence", 0.0),
                            frame_count=metadata.get("frame_count", 1),
                        )
                    except Exception as track_err:
                        logger.warning(
                            f"Failed to store face track: {track_err}"
                        )
            except Exception as e:
                logger.warning(f"Track finalization failed: {e}")

        # Build and Store Scenelets (Temporal Sequence Indexing)
        try:
            scenelets = scenelet_builder.build_scenelets()
            logger.info(
                f"Building {len(scenelets)} temporal scenelets for {path.name}..."
            )

            for sl in scenelets:
                self.db.store_scenelet(
                    media_path=str(path),
                    start_time=sl.start_ts,
                    end_time=sl.end_ts,
                    content_text=sl.fused_content,
                    payload={
                        "entities": sl.all_entities,
                        "actions": sl.all_actions,
                        "audio_text": sl.audio_text,
                    },
                )
            logger.info(f"Stored {len(scenelets)} scenelets successfully.")
        except Exception as e:
            logger.warning(f"Scenelet build/store failed: {e}")

        # Final cleanup
        del self.vision
        del self.faces
        self.vision = None
        self.faces = None
        if hasattr(self, "_face_track_builder"):
            del self._face_track_builder
        self._cleanup_memory()

    @observe("scene_captioning")
    async def _process_scene_captions(
        self, path: Path, job_id: str | None = None
    ) -> None:
        """Processes scene boundaries and aggregates multi-modal data.

        Identifies scene changes, creates visual summaries for each scene
        using VLM, aggregates dialogue and frame-level entities/faces,
        and stores everything with multi-vector embeddings for hybrid search.

        Args:
            path: Path to the media file.
            job_id: Optional ID for progress tracking.
        """
        scenes = detect_scenes(path)
        if not scenes:
            logger.info(f"No scene boundaries detected in {path.name}")
            return

        # Import aggregator
        from core.processing.scene_aggregator import aggregate_frames_to_scene

        vlm = get_vlm_client()
        prompt = (
            "Describe this scene in detail: actions, objects, colors, expressions, "
            "and atmosphere. Be specific about what people are doing and wearing."
        )

        # Get all audio segments for this video (for dialogue per scene)
        audio_segments = self._get_audio_segments_for_video(str(path))

        # Get all frames for this video (for aggregation per scene)
        all_frames = self._get_frames_for_video(str(path))

        scenes_stored = 0
        for idx, scene in enumerate(scenes):
            if job_id and progress_tracker.is_cancelled(job_id):
                break

            # 1. Get VLM caption for representative frame
            visual_summary = ""
            frame_bytes = extract_scene_frame(path, scene.mid_time)
            if frame_bytes:
                try:
                    caption = vlm.generate_caption_from_bytes(
                        frame_bytes, prompt
                    )
                    if caption:
                        visual_summary = caption
                except Exception as e:
                    logger.warning(f"VLM caption failed for scene {idx}: {e}")

            # 2. Filter frames that belong to this scene
            scene_frames = [
                f
                for f in all_frames
                if scene.start_time <= f.get("timestamp", 0) <= scene.end_time
            ]

            # 3. Filter audio segments that overlap this scene
            scene_audio = [
                a
                for a in audio_segments
                if a.get("end", 0) > scene.start_time
                and a.get("start", 0) < scene.end_time
            ]

            # 4. Aggregate frame data into scene
            aggregated = aggregate_frames_to_scene(
                frames=scene_frames,
                start_time=scene.start_time,
                end_time=scene.end_time,
                dialogue_segments=scene_audio,
            )

            # Override with VLM summary if available
            if visual_summary:
                aggregated["visual_summary"] = visual_summary

            # 5. Build text for each vector
            # Visual: entities, clothing, people
            visual_parts = []
            for name in aggregated.get("person_names", []):
                visual_parts.append(name)
            for attr in aggregated.get("person_attributes", []):
                if attr.get("clothing_color") and attr.get("clothing_type"):
                    visual_parts.append(
                        f"{attr['clothing_color']} {attr['clothing_type']}"
                    )
                visual_parts.extend(attr.get("accessories", []))
            for entity in aggregated.get("entities", []):
                if isinstance(entity, dict):
                    visual_parts.append(entity.get("name", ""))
            visual_parts.extend(aggregated.get("visible_text", []))
            if visual_summary:
                visual_parts.append(visual_summary)
            visual_text = " ".join(filter(None, visual_parts))

            # Motion: actions
            motion_parts = aggregated.get("actions", [])
            if aggregated.get("action_sequence"):
                motion_parts.append(aggregated["action_sequence"])
            motion_text = " ".join(filter(None, motion_parts))

            # Dialogue: transcript
            dialogue_text = aggregated.get("dialogue_transcript", "")

            # 6. Store scene with multi-vector
            try:
                # Save representative thumbnail
                thumb_path = None
                if frame_bytes:
                    import hashlib

                    thumb_dir = settings.cache_dir / "thumbnails" / "scenes"
                    thumb_dir.mkdir(parents=True, exist_ok=True)
                    safe_stem = hashlib.md5(path.stem.encode()).hexdigest()
                    thumb_name = f"{safe_stem}_{scene.start_time:.2f}.jpg"
                    thumb_file = thumb_dir / thumb_name
                    with open(thumb_file, "wb") as f:
                        f.write(frame_bytes)
                    thumb_path = f"/thumbnails/scenes/{thumb_name}"

                # Build payload
                payload = {
                    # Indexed for filtering
                    "face_cluster_ids": aggregated.get("face_cluster_ids", []),
                    "person_names": aggregated.get("person_names", []),
                    "clothing_colors": [
                        a.get("clothing_color", "")
                        for a in aggregated.get("person_attributes", [])
                        if a.get("clothing_color")
                    ],
                    "clothing_types": [
                        a.get("clothing_type", "")
                        for a in aggregated.get("person_attributes", [])
                        if a.get("clothing_type")
                    ],
                    "accessories": [
                        acc
                        for a in aggregated.get("person_attributes", [])
                        for acc in a.get("accessories", [])
                    ],
                    "actions": aggregated.get("actions", []),
                    "visible_text": aggregated.get("visible_text", []),
                    "entity_names": [
                        e.get("name", "")
                        for e in aggregated.get("entities", [])
                        if isinstance(e, dict)
                    ],
                    # Non-indexed data
                    "visual_summary": visual_summary,
                    "action_sequence": aggregated.get("action_sequence", ""),
                    "location": aggregated.get("location", ""),
                    "cultural_context": aggregated.get("cultural_context", ""),
                    "dialogue_transcript": dialogue_text,
                    "frame_count": aggregated.get("frame_count", 0),
                    "thumbnail_path": thumb_path,
                }

                self.db.store_scene(
                    media_path=str(path),
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    visual_text=visual_text,
                    motion_text=motion_text,
                    dialogue_text=dialogue_text,
                    payload=payload,
                )
                scenes_stored += 1

            except Exception as e:
                logger.warning(f"Failed to store scene {idx}: {e}")

        logger.info(
            f"Stored {scenes_stored}/{len(scenes)} scenes for {path.name}"
        )

    @observe("frame")
    async def _process_single_frame(
        self,
        *,
        video_path: Path,
        frame_path: Path,
        timestamp: float,
        index: int,
        context: str | None = None,
        neighbor_timestamps: list[float] | None = None,
    ) -> str | None:
        """Processes a single frame for identities and visual description.

        Performs face detection first to establish identity links, then runs
        structural vision analysis. Stores results in the database with
        linked face/voice info and temporal context.

        Args:
            video_path: Path to the source video.
            frame_path: Path to the extracted frame image.
            timestamp: Timestamp of the frame in the video.
            index: Sequential index of the frame.
            context: Narrative context from previous frames for VLM.
            neighbor_timestamps: Timestamps of neighboring frames for search.

        Returns:
            The generated frame description or None if processing failed.
        """
        if not self.vision or not self.faces:
            return None

        # 1. DETECT FACES FIRST (Capture Identity before Vision)
        face_cluster_ids: list[int] = []
        detected_faces = []
        try:
            detected_faces = await self.faces.detect_faces(frame_path)
        except Exception:
            pass

        # Feed detected faces into FaceTrackBuilder for temporal grouping
        # This creates stable per-video tracks before global identity linking
        if hasattr(self, "_face_track_builder") and detected_faces:
            self._face_track_builder.process_frame(
                faces=detected_faces,
                frame_index=index,
                timestamp=timestamp,
            )

        # Save face thumbnails
        thumb_dir = settings.cache_dir / "thumbnails" / "faces"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        # Create a safe file prefix using hash of the filename
        import hashlib

        safe_stem = hashlib.md5(video_path.stem.encode()).hexdigest()

        for idx, face in enumerate(detected_faces):
            if face.embedding is not None:
                # GLOBAL IDENTITY: Match against DB clusters (gallery-probe)
                # FaceManager.global_clusters loaded at _process_frames init
                cluster_id, self.faces.global_clusters = (
                    self.faces.match_or_create_cluster(
                        embedding=face.embedding,
                        existing_clusters=self.faces.global_clusters,
                        threshold=settings.face_clustering_threshold,
                    )
                )
                face_cluster_ids.append(cluster_id)

                # Crop and save face thumbnail with better quality
                thumb_path: str | None = None
                try:
                    import cv2
                    import numpy as np

                    if not frame_path.exists():
                        logger.warning(
                            f"[Thumb] Frame file missing: {frame_path}"
                        )
                    else:
                        img_data = np.fromfile(str(frame_path), dtype=np.uint8)
                        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                        if img is None:
                            logger.warning(
                                f"[Thumb] cv2.imdecode failed for {frame_path}"
                            )
                        else:
                            top, right, bottom, left = face.bbox
                            face_w = right - left
                            face_h = bottom - top

                            # More generous padding (25%) for better face context
                            pad_w = int(face_w * 0.25)
                            pad_h = int(face_h * 0.25)

                            # Less aggressive upward shift
                            shift_up = int(face_h * 0.05)

                            y1 = max(0, top - pad_h - shift_up)
                            y2 = min(img.shape[0], bottom + pad_h - shift_up)
                            x1 = max(0, left - pad_w)
                            x2 = min(img.shape[1], right + pad_w)

                            face_crop = img[y1:y2, x1:x2]

                            # Larger minimum size for HD quality
                            min_size = 256
                            crop_h, crop_w = face_crop.shape[:2]

                            if crop_h > 10 and crop_w > 10:
                                if crop_h < min_size or crop_w < min_size:
                                    scale = max(
                                        min_size / crop_h, min_size / crop_w
                                    )
                                    new_w = int(crop_w * scale)
                                    new_h = int(crop_h * scale)
                                    face_crop = cv2.resize(
                                        face_crop,
                                        (new_w, new_h),
                                        interpolation=cv2.INTER_LANCZOS4,
                                    )

                                # Use slightly lower quality (95) to save memory/space, explicit int cast for resize
                                thumb_name = (
                                    f"{safe_stem}_{timestamp:.2f}_{idx}.jpg"
                                )
                                thumb_file = thumb_dir / thumb_name

                                try:
                                    # Pre-check memory availability or just robust try/catch
                                    cv2.imwrite(
                                        str(thumb_file),
                                        face_crop,
                                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                                    )
                                    if (
                                        thumb_file.exists()
                                        and thumb_file.stat().st_size > 0
                                    ):
                                        thumb_path = (
                                            f"/thumbnails/faces/{thumb_name}"
                                        )
                                        logger.debug(
                                            f"[Thumb] Created: {thumb_path}"
                                        )
                                    else:
                                        logger.warning(
                                            f"Thumbnail file empty/missing after write: {thumb_file}"
                                        )
                                except cv2.error as e:
                                    logger.warning(
                                        f"Thumbnail save skipped (OpenCV error): {e}"
                                    )
                                    # Clean up if partial write happened
                                    if (
                                        thumb_file.exists()
                                        and thumb_file.stat().st_size == 0
                                    ):
                                        thumb_file.unlink()

                except (MemoryError, cv2.error) as e:
                    logger.warning(
                        f"Thumbnail generation skipped (OOM/CV error): {e}"
                    )
                    gc.collect()  # Try to recover leaks
                except Exception as e:
                    logger.error(f"Thumbnail generation failed: {e}")

                # Store face with PROPER cluster_id (not hash-based)
                self.db.insert_face(
                    face.embedding,
                    name=None,
                    cluster_id=cluster_id,  # Use proper cluster ID
                    media_path=str(video_path),
                    timestamp=timestamp,
                    thumbnail_path=thumb_path,
                    # Quality metrics for clustering
                    bbox_size=getattr(face, "_bbox_size", None),
                    det_score=face.confidence
                    if hasattr(face, "confidence")
                    else None,
                )

        # 2. RUN VISION ANALYSIS (With OCR and structured output)
        description: str | None = None
        analysis = None

        # CRITICAL: Unload face models to free VRAM for Ollama
        if self.faces:
            self.faces.unload_gpu()

        # Build identity context from HITL names for VLM
        identity_parts = []
        for idx, cid in enumerate(face_cluster_ids):
            name = self.db.get_face_name_by_cluster(cid)
            if name:
                identity_parts.append(f"Person {idx + 1}: {name}")
            else:
                identity_parts.append(
                    f"Person {idx + 1}: Unknown (cluster {cid})"
                )

        # Get speaker name at this timestamp
        try:
            speaker_clusters = self._get_speaker_clusters_at_time(
                str(video_path), timestamp
            )
            for scid in speaker_clusters:
                sname = self.db.get_speaker_name_by_cluster(scid)
                if sname:
                    identity_parts.append(f"Speaking: {sname}")
        except Exception:
            pass

        identity_context = "\n".join(identity_parts) if identity_parts else None

        try:
            # GPU-first: Unload GPU models before Ollama vision call to prevent OOM
            from core.utils.hardware import cleanup_vram, log_vram_status

            if self.faces:
                self.faces.unload_gpu()
            cleanup_vram()
            log_vram_status("before_ollama")

            # Build video context to prevent VLM hallucinations
            # CRITICAL: No hardcoded filename patterns - purely content-based
            # Content type is determined by:
            # 1. HITL override (user explicitly set via ingestion param)
            # 2. Audio classification (music % vs speech %)
            # 3. Fallback to neutral context
            video_context_parts = [
                f"Filename: {video_path.stem}",
            ]

            # Check for HITL content type override (set via ingestion API)
            hitl_content_type = getattr(self, "_hitl_content_type", None)
            audio_classification = getattr(self, "_audio_classification", None)

            if hitl_content_type:
                # User explicitly set content type during ingestion
                video_context_parts.append(
                    f"Content Type (User Override): {hitl_content_type}"
                )
                if hitl_content_type.lower() in (
                    "song",
                    "music",
                    "music_video",
                ):
                    video_context_parts.append(
                        "INSTRUCTION: This is a music video. Describe choreography and visuals, NOT imaginary conversations."
                    )
                elif hitl_content_type.lower() in (
                    "interview",
                    "podcast",
                    "talk",
                ):
                    video_context_parts.append(
                        "INSTRUCTION: This is conversational content. Describe the speakers and their expressions."
                    )
            elif audio_classification:
                # Content-based detection from audio analysis
                music_pct = audio_classification.get("music_percentage", 0)
                speech_pct = audio_classification.get("speech_percentage", 0)

                if music_pct > 70:
                    video_context_parts.append(
                        f"Audio Analysis: {music_pct:.0f}% music, {speech_pct:.0f}% speech"
                    )
                    video_context_parts.append(
                        "INSTRUCTION: High music content detected. Focus on visuals and movement, not dialogue."
                    )
                elif speech_pct > 70:
                    video_context_parts.append(
                        f"Audio Analysis: {speech_pct:.0f}% speech, {music_pct:.0f}% music"
                    )
                    video_context_parts.append(
                        "INSTRUCTION: High speech content detected. Describe speakers and context."
                    )
                else:
                    video_context_parts.append(
                        f"Audio Analysis: Mixed content ({music_pct:.0f}% music, {speech_pct:.0f}% speech)"
                    )

            # Always add grounding rules regardless of content type
            video_context_parts.append("")
            video_context_parts.append("GROUNDING RULES (ALWAYS FOLLOW):")
            video_context_parts.append(
                "1. Describe ONLY what you SEE in the frame"
            )
            video_context_parts.append(
                "2. Do NOT hallucinate conversations, events, or contexts not visible"
            )
            video_context_parts.append(
                "3. Be specific about actions, clothing, and objects visible"
            )

            video_context = "\n".join(video_context_parts)

            # ============================================================
            # OCR WIRING FIX: Previously dead code - now actually called!
            # Uses text_gate to avoid running expensive OCR on frames without text
            # ============================================================
            ocr_text = ""
            ocr_boxes = []
            try:
                # Load frame as numpy array (Windows path safe)
                frame_data = np.fromfile(str(frame_path), dtype=np.uint8)
                frame_img = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

                if frame_img is not None:
                    # Gate: Only run OCR if frame likely contains text (edge density check)
                    if self.text_gate.has_text(frame_img):
                        ocr_result = await self.ocr_engine.extract_text(
                            frame_img
                        )
                        if ocr_result and ocr_result.get("text"):
                            ocr_text = ocr_result["text"]
                            ocr_boxes = ocr_result.get("boxes", [])
                            logger.info(f"[OCR] Extracted: {ocr_text[:100]}...")
                        else:
                            logger.debug("[OCR] No text found in gated frame")
            except Exception as e:
                logger.warning(f"[OCR] Failed: {e}")

            # Run structured vision analysis
            analysis = await self.vision.analyze_frame(
                frame_path,
                video_context=video_context,
                identity_context=identity_context,
                temporal_context=context,
            )
            if analysis:
                description = analysis.to_search_content()
                analysis.face_ids = [str(cid) for cid in face_cluster_ids]
        except Exception as e:
            logger.warning(
                f"Structured analysis failed: {e}, falling back to describe"
            )

        # Fallback to unstructured description
        if not description:
            try:
                description = await self.vision.describe(
                    frame_path, context=context
                )
            except Exception:
                pass

        if description:
            vector = self.db.encode_texts(description)[0]

            # Build structured payload for accurate search with filterable fields
            payload: dict[str, Any] = {
                "face_cluster_ids": face_cluster_ids,
                "face_names": [],
                "speaker_names": [],
            }

            # 3a. FACE-AUDIO MAPPING: Link faces to speakers at this timestamp
            # Get speaker name if someone is speaking at this frame's timestamp
            try:
                speaker_cluster_ids = self._get_speaker_clusters_at_time(
                    media_path=str(video_path),
                    timestamp=timestamp,
                )

                # Bi-directional Name Propagation
                # If we have a named Face and an unnamed Speaker -> Name the Speaker
                # If we have a named Speaker and an unnamed Face -> Name the Face

                # First, gather face names for this frame
                current_face_names = {}  # cluster_id -> name
                for cid in face_cluster_ids:
                    fname = self.db.get_face_name_by_cluster(cid)
                    if fname:
                        current_face_names[cid] = fname
                        payload["face_names"].append(fname)

                # Now process speaker clusters
                for cluster_id in speaker_cluster_ids:
                    speaker_name = self.db.get_speaker_name_by_cluster(
                        cluster_id
                    )

                    if speaker_name:
                        payload["speaker_names"].append(speaker_name)
                        # Propagate Speaker Name -> Unnamed Faces
                        if not current_face_names and face_cluster_ids:
                            # Heuristic: If there's exactly one unnamed face and one named speaker, link them
                            if len(face_cluster_ids) == 1:
                                face_cid = face_cluster_ids[0]
                                logger.info(
                                    f"Auto-mapping Speaker '{speaker_name}' -> Face Cluster {face_cid}"
                                )
                                self.db.set_face_name(face_cid, speaker_name)
                                payload["face_names"].append(
                                    speaker_name
                                )  # Update current payload

                    elif current_face_names:
                        # Propagate Face Name -> Unnamed Speaker
                        # Heuristic: If exactly one named face is visible, assume they are the speaker
                        if len(current_face_names) == 1:
                            face_name = next(iter(current_face_names.values()))
                            logger.info(
                                f"Auto-mapping Face '{face_name}' -> Speaker Cluster {cluster_id}"
                            )
                            self.db.set_speaker_name(cluster_id, face_name)
                            payload["speaker_names"].append(
                                face_name
                            )  # Update current payload

            except Exception as e:
                logger.warning(f"Face-Audio mapping error: {e}")

            # 3b. (Skipped redundant loop, handled above)

            # 3c. Build identity text for searchability
            identity_parts = []
            if payload["face_names"]:
                identity_parts.append(
                    f"Visible: {', '.join(payload['face_names'])}"
                )
            if payload["speaker_names"]:
                identity_parts.append(
                    f"Speaking: {', '.join(payload['speaker_names'])}"
                )
            if identity_parts:
                payload["identity_text"] = ". ".join(identity_parts)

            # Add structured data if available for hybrid search
            if analysis:
                payload["structured_data"] = analysis.model_dump()
                payload["visible_text"] = (
                    analysis.scene.visible_text if analysis.scene else []
                )
                payload["entities"] = (
                    [e.name for e in analysis.entities]
                    if analysis.entities
                    else []
                )
                payload["entity_categories"] = (
                    list({e.category for e in analysis.entities})
                    if analysis.entities
                    else []
                )
                payload["scene_location"] = (
                    analysis.scene.location if analysis.scene else ""
                )
                payload["action"] = analysis.action or ""
                payload["description"] = description

            # 3d. Add structured face data for UI overlays (bboxes)
            # This enables drawing boxes around identified people in the UI
            faces_metadata = []
            for face, cluster_id in zip(
                detected_faces, face_cluster_ids, strict=False
            ):
                face_name = self.db.get_face_name_by_cluster(cluster_id)
                faces_metadata.append(
                    {
                        "bbox": face.bbox
                        if isinstance(face.bbox, list)
                        else list(face.bbox),  # [top, right, bottom, left]
                        "cluster_id": cluster_id,
                        "name": face_name,
                        "confidence": face.confidence,
                    }
                )

            if faces_metadata:
                payload["faces"] = faces_metadata

            # ============================================================
            # OCR BOXES: Store bounding boxes for UI overlay
            # Enables drawing text regions on video player
            # ============================================================
            if ocr_text:
                payload["ocr_text"] = ocr_text
            if ocr_boxes:
                payload["ocr_boxes"] = ocr_boxes

            # 3e. Add temporal context for video-aware search (not isolated frames)
            # This enables queries like "pin falling slowly" by connecting adjacent frames
            if neighbor_timestamps:
                payload["neighbor_timestamps"] = neighbor_timestamps
                payload["temporal_window_size"] = len(neighbor_timestamps)
            if context:
                # Store truncated context for retrieval boost
                payload["temporal_context"] = (
                    context[:500] if len(context) > 500 else context
                )

            # 3f. Generate Vector (Include identity for searchability)
            # "A man walking. Visible: John. Speaking: John"
            full_text = description
            if "identity_text" in payload:
                full_text = f"{description}. {payload['identity_text']}"

            vector = self.db.encode_texts(full_text)[0]

            # Generate a proper UUID for Qdrant (file paths are not valid point IDs)
            import uuid

            frame_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{video_path}_{timestamp:.3f}")
            )

            self.db.upsert_media_frame(
                point_id=frame_id,
                vector=vector,
                video_path=str(video_path),
                timestamp=timestamp,
                action=description,
                payload=payload,
                ocr_text=ocr_text,  # Add extracted text
            )

        return description

    def _get_speaker_clusters_at_time(
        self, media_path: str, timestamp: float
    ) -> list[int]:
        """Identifies speaker cluster IDs active at a specific timestamp.

        Enables cross-modal mapping by finding which speaker is talking
        at the moment a particular face or visual event occurs.

        Args:
            media_path: Path to the media file.
            timestamp: The timestamp in seconds.

        Returns:
            A list of active speaker cluster IDs.
        """
        clusters = []
        try:
            # Query voice segments that overlap with this timestamp
            voice_segments = self.db.get_voice_segments_for_media(media_path)
            for seg in voice_segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                if start <= timestamp <= end:
                    cluster_id = seg.get("cluster_id")
                    if cluster_id is not None and cluster_id not in clusters:
                        clusters.append(cluster_id)
        except Exception:
            pass
        return clusters

    def _get_audio_segments_for_video(self, media_path: str) -> list[dict]:
        """Retrieves all dialogue/audio segments for a specific video.

        Querying the database for previously stored segments to be used
        in high-level aggregation or scene summarization.

        Args:
            media_path: Path to the media file.

        Returns:
            A list of audio segment dictionaries containing start, end, and text.
        """
        try:
            # Query media_segments collection for this video
            resp = self.db.client.scroll(
                collection_name=self.db.MEDIA_SEGMENTS_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_path",
                            match=models.MatchValue(value=media_path),
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
            )
            return [
                {
                    "start": p.payload.get("start", 0),
                    "end": p.payload.get("end", 0),
                    "text": p.payload.get("text", ""),
                    "type": p.payload.get("type", "dialogue"),
                }
                for p in resp[0]
                if p.payload
            ]
        except Exception:
            return []

    def _get_frames_for_video(self, media_path: str) -> list[dict]:
        """Retrieves all sampled and analyzed frames for a video.

        Provides a consolidated view of visual detections (faces, entities,
        actions) across the video timeline for scene-level reasoning.

        Args:
            media_path: Path to the media file.

        Returns:
            A list of frame data dictionaries ordered by timestamp.
        """
        try:
            # Query media_frames collection for this video
            resp = self.db.client.scroll(
                collection_name=self.db.MEDIA_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_path",
                            match=models.MatchValue(value=media_path),
                        )
                    ]
                ),
                limit=5000,  # Allow more frames for long videos
                with_payload=True,
            )
            frames = []
            for p in resp[0]:
                if p.payload:
                    frames.append(
                        {
                            "id": str(p.id),
                            "timestamp": p.payload.get("timestamp", 0),
                            "action": p.payload.get("action", ""),
                            "description": p.payload.get("description", "")
                            or p.payload.get("action", ""),
                            "face_cluster_ids": p.payload.get(
                                "face_cluster_ids", []
                            ),
                            "face_names": p.payload.get("face_names", []),
                            "speaker_names": p.payload.get("speaker_names", []),
                            "visible_text": p.payload.get("visible_text", []),
                            "entities": p.payload.get("entities", []),
                            "structured_data": p.payload.get(
                                "structured_data", {}
                            ),
                            "scene_location": p.payload.get(
                                "scene_location", ""
                            ),
                            "scene_cultural": p.payload.get(
                                "scene_cultural", ""
                            ),
                        }
                    )
            # Sort by timestamp
            frames.sort(key=lambda x: x.get("timestamp", 0))
            return frames
        except Exception:
            return []

    def _prepare_segments_for_db(
        self,
        *,
        path: Path,
        chunks: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prepares raw transcription chunks for ingestion into the database.

        Filters empty segments and ensures consistent timing metadata.

        Args:
            path: Path to the media file.
            chunks: Iterable of raw segment dictionaries.

        Returns:
            A list of cleaned and formatted segment dictionaries.
        """
        prepared: list[dict[str, Any]] = []
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            start = chunk.get("start")
            end = chunk.get("end")
            if start is None:
                continue
            if end is None:
                end = float(start) + 2.0
            prepared.append(
                {
                    "text": text,
                    "start": float(start),
                    "end": float(end),
                    "type": "dialogue",
                }
            )
        return prepared

    async def _post_process_video(self, path: Path, job_id: str) -> None:
        """Executes global enrichment and cross-modal linking after ingestion.

        Finalizes global context, stores high-level summaries, attaches
        thumbnails, and performs deep video tracking (SAM 3).

        Args:
            path: Path to the media file.
            job_id: The ID of the ingestion job.
        """
        media_path = str(path)

        try:
            from core.processing.scene_aggregator import GlobalContextManager

            global_ctx = GlobalContextManager()
            frames = self._get_frames_for_video(media_path)
            audio_segments = self._get_audio_segments_for_video(media_path)

            # Deep Video Understanding: SAM 3 Concept Tracking
            # Only run if enabled and frames exist
            if self.sam3_tracker and frames:
                try:
                    self._process_video_masklets(path, frames)
                except Exception as e:
                    logger.warning(f"SAM3 Tracking failed: {e}")

            if frames:
                # Collect dialogue from audio segments
                dialogue_texts = [
                    seg.get("text", "")
                    for seg in audio_segments
                    if seg.get("text")
                ]
                dialogue_summary = " ".join(
                    dialogue_texts[:50]
                )  # First 50 segments

                scene_data = {
                    "start_time": 0,
                    "end_time": frames[-1].get("timestamp", 0) + 1,
                    "visual_summary": " ".join(
                        f.get("action", "") for f in frames[:20]
                    ),
                    "person_names": list(
                        {n for f in frames for n in f.get("face_names", [])}
                    ),
                    "location": frames[0].get("scene_location", "")
                    if frames
                    else "",
                    "entities": [
                        e for f in frames for e in f.get("entities", [])
                    ],
                    "dialogue_summary": dialogue_summary,
                }
                global_ctx.add_scene(scene_data)

                global_summary = global_ctx.to_payload()

                # Generate and attach Main Video Thumbnail
                try:
                    main_thumb = self._generate_main_thumbnail(path)
                    if main_thumb:
                        global_summary["thumbnail_path"] = main_thumb
                except Exception as e:
                    logger.warning(f"Thumbnail attachment failed: {e}")

                logger.info(
                    f"[PostProcess] Global context: {global_summary.get('scene_count', 0)} scenes, top people: {global_summary.get('top_people', [])[:3]}"
                )

                try:
                    self.db.update_video_metadata(
                        media_path, metadata=global_summary
                    )
                except Exception as e:
                    logger.warning(
                        f"[PostProcess] Failed to update global context: {e}"
                    )
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")

    def _generate_main_thumbnail(self, path: Path) -> str | None:
        """Generates a representative thumbnail for the video at 5.0s.

        Args:
            path: Path to the media file.

        Returns:
            The relative web path to the generated thumbnail, or None on failure.
        """
        import hashlib
        import subprocess

        try:
            thumb_dir = settings.cache_dir / "thumbnails" / "videos"
            thumb_dir.mkdir(parents=True, exist_ok=True)

            safe_stem = hashlib.md5(path.stem.encode()).hexdigest()
            thumb_name = f"{safe_stem}_main.jpg"
            thumb_file = thumb_dir / thumb_name

            # Return relative path for frontend
            rel_path = f"/thumbnails/videos/{thumb_name}"

            if thumb_file.exists():
                return rel_path

            # Extract at 5 seconds
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                "00:00:05.000",
                "-i",
                str(path),
                "-vframes",
                "1",
                "-q:v",
                "2",
                str(thumb_file),
            ]

            # Run ffmpeg
            subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Fallback to 0s if 5s failed (e.g. short video)
            if not thumb_file.exists():
                cmd[3] = "00:00:00.000"
                subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            if thumb_file.exists():
                return rel_path
            return None

        except Exception as e:
            logger.warning(f"Failed to generate main thumbnail: {e}")
            return None

    def _process_video_masklets(self, path: Path, frames: list[dict]) -> None:
        """Executes Segment-Anything-2 (SAM 3) tracking for top visual concepts.

        Identifies recurring or unique entities across frames and generates
        spatio-temporal tracking data (masklets) for precise retrieval.

        Args:
            path: Path to the media file.
            frames: List of already analyzed frame metadata.
        """
        # 1. Extract potential concepts from frame entities/descriptions
        concept_counts = {}
        for f in frames:
            # Entities
            for e in f.get("entities", []):
                concept_counts[e] = concept_counts.get(e, 0) + 1
            # Keywords from action (simple heuristic)
            action = f.get("action", "")
            if "holding a" in action:
                try:
                    obj = (
                        action.split("holding a")[1]
                        .split()[0]
                        .strip()
                        .strip(".,")
                    )
                    if len(obj) > 2:
                        concept_counts[obj] = concept_counts.get(obj, 0) + 1
                except Exception:
                    pass

        # 2. Select top 5 concepts to track
        top_concepts = sorted(
            concept_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        prompts = [c[0] for c in top_concepts]

        if not prompts:
            logger.info("No concepts found to track with SAM3.")
            return

        logger.info(f"SAM3 Tracking Concepts: {prompts}")

        # 3. Run Tracker
        # Result aggregation: TrackID -> {start, end, max_conf}
        tracks = {}

        # SAM3 returns iterator of {frame_idx, object_ids, masks}
        # object_ids maps to the index in 'prompts' list added sequentially?
        # Actually Sam3Tracker.add_concept_prompt adds one text.
        # We need to map object_id back to prompt text.
        # Implementation Detail: Sam3 wrapper doesn't provide easy mapping back yet.
        # We will iterate prompts and run sequentially or concurrently if supported.
        # Sam3Tracker.process_video_concepts runs all prompts.
        # The object IDs returned correspond to sequential addition.
        # i.e. Prompt 0 -> obj_id 0, Prompt 1 -> obj_id 1 (usually).

        # We assume 1-to-1 for now.

        if self.sam3_tracker:
            for frame_data in self.sam3_tracker.process_video_concepts(
                path, prompts
            ):
                frame_idx = frame_data["frame_idx"]
                obj_ids = frame_data["object_ids"]

                for obj_id in obj_ids:
                    # Get concept name
                    if obj_id < len(prompts):
                        concept = prompts[obj_id]
                    else:
                        concept = f"object_{obj_id}"

                    track_key = f"{concept}_{obj_id}"

                if track_key not in tracks:
                    tracks[track_key] = {
                        "start": frame_idx,
                        "end": frame_idx,
                        "concept": concept,
                    }
                else:
                    tracks[track_key]["end"] = max(
                        tracks[track_key]["end"], frame_idx
                    )

        # 4. Save Masklets to DB
        fps = (
            settings.frame_interval
        )  # Ingestion loop uses frame_interval approx?
        # Actually frames have timestamps. We can map frame_idx to timestamp roughly.
        # Or better: pipeline knows fps or duration.
        # We can map frame_idx to time if we know video FPS.
        # For now, we estimate based on frame_interval setting if available, or just index.
        # Ideally we should use CV2 to get FPS of source to map frame_idx -> time.

        # Simpler: Use frame data if we have it? No, SAM3 processes all frames.
        # We will assume standard 30fps for timestamp estimation if metadata unavailable,
        # or fetch it.
        import cv2

        try:
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        except Exception:
            fps = 30.0

        for _key, data in tracks.items():
            start_time = data["start"] / fps
            end_time = data["end"] / fps
            duration = end_time - start_time

            if duration > 0.5:  # Ignore blips
                self.db.insert_masklet(
                    video_path=str(path),
                    concept=data["concept"],
                    start_time=start_time,
                    end_time=end_time,
                    confidence=0.9,  # SAM3 is usually confident
                )
                logger.info(
                    f"Masklet saved: {data['concept']} ({start_time:.1f}-{end_time:.1f}s)"
                )
