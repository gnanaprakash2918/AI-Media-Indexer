"""Media ingestion pipeline orchestrator."""

from __future__ import annotations

import asyncio
import gc
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import numpy as np
from qdrant_client.http import models

from config import settings
from core.llm.vlm_factory import get_vlm_client
from core.processing.deep_research import get_deep_research_processor
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
from core.utils.resource_arbiter import RESOURCE_ARBITER
from core.utils.retry import retry
from core.errors import (
    MediaIndexerError,
    IngestionError
)
import traceback

# Global semaphore for VLM parallelism (limits concurrent VLM calls to 4)
# This prevents OOM when using Ollama/Gemini with large contexts
VLM_SEMAPHORE = asyncio.Semaphore(4)



class FrameBuffer:
    """Buffers frame data for batch database writes.
    
    Accumulates processed frame data and flushes to DB in batches
    of `batch_size` for ~10x performance over individual writes.
    """
    
    def __init__(self, db: VectorDB, batch_size: int = 50):
        self.db = db
        self.batch_size = batch_size
        self._buffer: list[dict] = []
        self._total_flushed = 0
    
    def add(self, frame_data: dict) -> int:
        """Add a frame to the buffer. Returns frames flushed (0 or batch_size)."""
        self._buffer.append(frame_data)
        if len(self._buffer) >= self.batch_size:
            return self.flush()
        return 0
    
    def flush(self) -> int:
        """Flush all buffered frames to database."""
        if not self._buffer:
            return 0
        count = self.db.upsert_media_frames_batch(self._buffer)
        self._total_flushed += count
        self._buffer.clear()
        return count
    
    @property
    def pending(self) -> int:
        """Number of frames waiting to be flushed."""
        return len(self._buffer)
    
    @property
    def total_written(self) -> int:
        """Total frames written to database."""
        return self._total_flushed + len(self._buffer)


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
        self.prober = MediaProber()
        self.sam_tracker = Sam3Tracker()

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
        
        # Visual encoder for CLIP/SigLIP embeddings (lazy-loaded)
        self._visual_encoder = None

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
        """Orchestrates the full ingestion of a video file."""
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
            probed = await self.prober.probe(path)
            duration = float(probed.get("format", {}).get("duration", 0.0))
        except MediaProbeError as e:
            progress_tracker.fail(job_id, error=f"Media probe failed: {e}")
            raise

        # === CHUNKING DECISION (OOM Prevention) ===
        # Prevent OOM by processing long videos in chunks
        chunk_enabled = getattr(settings, 'enable_chunking', True)
        chunk_duration = getattr(settings, 'chunk_duration_seconds', 600)  # 10 min default
        min_length_for_chunk = getattr(settings, 'min_media_length_for_chunking', 1800)  # 30 min
        auto_chunk_hw = getattr(settings, 'auto_chunk_by_hardware', True)
        
        # Auto-adjust chunk size based on hardware
        if auto_chunk_hw and torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 8:
                    chunk_duration = min(chunk_duration, 300)  # 5 min for low VRAM
                    min_length_for_chunk = 600  # Chunk anything > 10 min
                    logger.info(f"[Chunking] Low VRAM ({vram_gb:.1f}GB) - using 5min chunks")
            except Exception:
                pass
        
        should_chunk = chunk_enabled and duration > min_length_for_chunk
        
        if should_chunk:
            logger.info(
                f"[Chunking] Video {duration/60:.1f}min > threshold {min_length_for_chunk/60:.0f}min. "
                f"Will process in {chunk_duration/60:.0f}min chunks."
            )
            # Store chunk info for _process_frames to use
            self._chunk_duration = chunk_duration
            self._total_chunks = int(duration / chunk_duration) + 1
        else:
            self._chunk_duration = None
            self._total_chunks = 1

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
                async with progress_tracker.stage(job_id, "audio", "Processing audio"):
                    progress_tracker.update(job_id, 10.0)
                    await retry(
                        lambda: self._process_audio(path),
                        on_retry=lambda e: progress_tracker.increment_retry(job_id, "audio")
                    )
                    progress_tracker.update(job_id, 30.0)
                
                logger.info(
                    "[Pipeline] _process_audio completed, running cleanup..."
                )
                self._cleanup_memory("audio_complete")  # Unload Whisper
                logger.info(
                    "[Pipeline] Audio cleanup done, saving checkpoint..."
                )
                # Checkpoint audio completion
                progress_tracker.save_checkpoint(
                    job_id, {"audio_complete": True}
                )
            else:
                progress_tracker.stage_start(job_id, "audio", "Skipped (Already done)")
                progress_tracker.stage_complete(job_id, "audio", "Skipped")
            logger.info(
                "[Pipeline] Audio phase complete, moving to voice processing..."
            )


            if progress_tracker.is_cancelled(job_id):
                return job_id
            if progress_tracker.is_paused(job_id):
                return job_id

            if not skip_voice:
                async with progress_tracker.stage(job_id, "voice", "Processing voice"):
                    progress_tracker.update(job_id, 35.0)
                    await retry(
                        lambda: self._process_voice(path),
                        on_retry=lambda e: progress_tracker.increment_retry(job_id, "voice")
                    )
                    progress_tracker.update(job_id, 50.0)
                
                self._cleanup_memory("voice_complete")  # Unload Pyannote
                # Checkpoint voice completion
                progress_tracker.save_checkpoint(
                    job_id, {"voice_complete": True}
                )
            else:
                 progress_tracker.stage_start(job_id, "voice", "Skipped (Already done)")
                 progress_tracker.stage_complete(job_id, "voice", "Skipped")

            logger.debug("Voice complete - checking job status")

            if progress_tracker.is_cancelled(job_id):
                logger.info(f"Job {job_id} cancelled")
                return job_id

            if progress_tracker.is_paused(job_id):
                logger.info(f"Job {job_id} paused")
                return job_id

            # Audio Events (CLAP)
            async with progress_tracker.stage(job_id, "audio_events", "Detecting audio events"):
                await self._process_audio_events(path, job_id)


            logger.debug("Starting frame processing")
            async with progress_tracker.stage(job_id, "frames", "Processing frames"):
                progress_tracker.update(job_id, 55.0)
                await retry(
                    lambda: self._process_frames(
                        path, job_id, total_duration=duration
                    ),
                    on_retry=lambda e: progress_tracker.increment_retry(job_id, "frames")
                )
                progress_tracker.update(job_id, 85.0)

            logger.debug("Frame processing complete - cleaning memory")
            self._cleanup_memory("frames_complete")

            if progress_tracker.is_cancelled(
                job_id
            ) or progress_tracker.is_paused(job_id):
                return job_id

            # Dense Scene Captioning (VLM on detected scene boundaries)
            async with progress_tracker.stage(job_id, "scene_captions", "Generating scene captions"):
                progress_tracker.update(job_id, 90.0)
                await self._process_scene_captions(path, job_id)

            # Post-Processing Phase
            async with progress_tracker.stage(job_id, "post_processing", "Enriching metadata"):
                progress_tracker.update(job_id, 95.0)
                await self._post_process_video(path, job_id)

            progress_tracker.complete(job_id, message=f"Completed: {path.name}")

            return job_id

        except MediaIndexerError as e:
            logger.error(f"Ingestion failed with known error: {e}")
            progress_tracker.fail(job_id, error=str(e))
            raise

        except Exception as e:
            logger.critical(f"Ingestion failed with UNEXPECTED error: {e}")
            logger.critical(traceback.format_exc())
            progress_tracker.fail(job_id, error=f"Unexpected: {e}")
            raise IngestionError(f"Unexpected pipeline failure: {e}", original_error=e)

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
                            probed = await self.prober.probe(path)
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

                    audio_segments = (
                        await indic_transcriber.transcribe(
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
                                    await transcriber.transcribe(
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
                    probed = await self.prober.probe(path)
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
            await self.db.insert_media_segments(str(path), prepared)
            log(f"[Audio] Stored {len(prepared)} dialogue segments in DB")

            try:
                probed = await self.prober.probe(path)
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
            # CLAP defaults to True in EnhancedPipelineConfig, so use True as fallback
            if self.enhanced_config and getattr(
                self.enhanced_config, "enable_clap", True
            ):
                from core.processing.audio_events import AudioEventDetector

                log("[CLAP] Starting audio event detection...")
                audio_detector = AudioEventDetector()

                # === STREAMING APPROACH (OOM FIX) ===
                # Instead of loading entire audio file at once, we stream in chunks
                # This prevents OOM on videos longer than 15 minutes
                try:
                    # Get duration via FFprobe (doesn't load file into memory)
                    import subprocess
                    probe_cmd = [
                        'ffprobe', '-v', 'quiet', '-show_entries',
                        'format=duration', '-of', 'csv=p=0', str(path)
                    ]
                    duration_str = await asyncio.to_thread(
                        lambda: subprocess.check_output(probe_cmd).decode().strip()
                    )
                    total_duration = float(duration_str)
                    log(f"[CLAP] Video duration: {total_duration:.1f}s (streaming mode)")
                    
                    # Streaming parameters
                    sr = 16000  # CLAP expected sample rate
                    chunk_duration = 30.0  # 30s chunks (~1MB RAM each)
                    stride = 25.0  # 5s overlap to catch boundary events
                    
                    # Prepare 2-second CLAP windows within each chunk
                    clap_window_duration = 2.0
                    clap_samples_per_window = int(clap_window_duration * sr)
                    
                    # Build list of (chunk, start_time) tuples for batch processing
                    audio_chunks: list[tuple] = []
                    
                    # Stream each chunk using FFmpeg (NOT librosa)
                    for chunk_start in range(0, int(total_duration), int(stride)):
                        chunk_end = min(chunk_start + chunk_duration, total_duration)
                        
                        # Use FFmpeg to extract just this chunk
                        chunk_array = await self._extract_audio_segment(
                            path, float(chunk_start), float(chunk_end), sr
                        )
                        
                        if chunk_array is None or len(chunk_array) < sr:
                            continue
                        
                        # Split chunk into 2-second CLAP windows  
                        for i in range(0, len(chunk_array) - clap_samples_per_window + 1, clap_samples_per_window):
                            window = chunk_array[i : i + clap_samples_per_window]
                            if len(window) < sr:  # Skip windows < 1 second
                                continue
                            window_start = chunk_start + (i / sr)
                            audio_chunks.append((window, window_start))
                        
                        # Aggressive cleanup after each chunk
                        del chunk_array
                        import gc
                        gc.collect()
                    
                    log(
                        f"[CLAP] Prepared {len(audio_chunks)} windows for batch processing (streaming mode)"
                    )

                    # Define target classes once
                    target_classes = [
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
                    ]

                    # BATCH PROCESSING: Single GPU acquisition for all chunks
                    # Run CLAP directly (it handles GPU locking internally)
                    batch_results = await audio_detector.detect_events_batch(
                        audio_chunks=audio_chunks,
                        target_classes=target_classes,
                        sample_rate=int(sr),
                        top_k=3,
                        threshold=0.25,
                    )

                    # Collect results with timestamps
                    audio_events = []
                    for (_, start_time), events in zip(
                        audio_chunks, batch_results
                    ):
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

            log("[Loudness] Starting audio level analysis (FFmpeg streaming)...")
            
            # === FFmpeg EBUR128 LOUDNESS ANALYSIS (OOM-SAFE) ===
            # Uses FFmpeg's ebur128 filter which streams the audio (~50MB RAM max)
            # instead of loading entire file into RAM (3-4GB for long videos)
            import subprocess
            import re
            
            try:
                # Run FFmpeg with ebur128 loudness filter
                cmd = [
                    "ffmpeg", "-i", str(path),
                    "-af", "ebur128=framelog=verbose:peak=true",
                    "-f", "null", "-"
                ]
                result = await asyncio.to_thread(
                    lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=300)
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
                estimated_spl = max(0, 85 + lufs + 23)  # 85dB baseline at -23 LUFS
                
                # Categorize
                if estimated_spl < 60:
                    category = "quiet"
                elif estimated_spl < 75:
                    category = "moderate"
                elif estimated_spl < 85:
                    category = "loud"
                else:
                    category = "very_loud"
                
                log(f"[Loudness] Overall: {estimated_spl:.0f} dB SPL ({category}) [LUFS: {lufs:.1f}]")
                
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
            from core.processing.audio_structure import get_music_analyzer

            log("[MusicStructure] Starting structure analysis...")
            music_analyzer = get_music_analyzer()

            # Load audio if not already loaded
            # SAFETY: Only load first 5 minutes for music structure to prevent OOM
            # Music structure (verse/chorus) is typically established early
            if "audio_array" not in locals():
                import librosa

                max_duration = 300.0  # 5 minutes max for music analysis
                audio_array, sr = librosa.load(
                    str(path), sr=22050, mono=True, duration=max_duration
                )
                log(f"[MusicStructure] Loaded {len(audio_array)/sr:.1f}s audio (limited to {max_duration}s)")
            else:
                # Resample to 22050 for librosa if needed
                if "sr" in locals() and sr != 22050:
                    import librosa

                    audio_array = librosa.resample(
                        audio_array, orig_sr=sr, target_sr=22050
                    )
                    sr = 22050

            # Analyze music structure
            analysis = music_analyzer.analyze_array(audio_array, sr=22050)

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

    async def _detect_audio_language_with_confidence(
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

        wav_path = None
        try:
            # Slice audio asynchronously in the main event loop
            from core.processing.transcriber import AudioTranscriber
            
            try:
                # Instantiate usage because _slice_audio is an instance method
                with AudioTranscriber() as transcriber:
                    wav_path = await transcriber._slice_audio(
                        path, start=start_offset, end=start_offset + duration
                    )
            except Exception as e:
                from core.utils.logger import log
                log(f"[Audio] Slicing failed: {e}, falling back to full file")
                wav_path = path

            # Run blocking detection in a thread
            return await asyncio.to_thread(
                self._run_detection_with_confidence,
                wav_path, # Pass the sliced audio (or original path)
            )

        except Exception as e:
            from core.utils.logger import log
            log(f"[Audio] Language detection failed: {e}")
            return ("en", 0.0)
            
        finally:
            # Cleanup temp file if created
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

    def _run_detection_with_confidence(
        self,
        wav_input: Path | bytes,
    ) -> tuple[str, float]:
        """Synchronous helper for language detection with confidence.

        Args:
            wav_input: Path to audio file or raw WAV bytes.

        Returns:
            Tuple of (language_code, confidence_score).
        """
        import io
        from core.utils.logger import log

        with AudioTranscriber() as transcriber:
            try:
                # Load model if needed
                model_id = "Systran/faster-whisper-base"
                if AudioTranscriber._SHARED_SIZE != model_id:
                    transcriber._load_model(model_id)

                if AudioTranscriber._SHARED_MODEL is None:
                    return ("en", 0.0)

                # Prepare input source
                if isinstance(wav_input, bytes):
                    input_file = io.BytesIO(wav_input)
                else:
                    input_file = str(wav_input)

                # Run detection on the sliced segment
                _, info = AudioTranscriber._SHARED_MODEL.transcribe(
                    input_file,
                    task="transcribe",
                    beam_size=5,
                )

                detected_lang = info.language or "en"
                confidence = info.language_probability or 0.0

                # Special handling for Indic languages with lower threshold
                indic_langs = [
                    "ta", "hi", "te", "ml", "kn", "bn", "gu", "mr", "or", "pa"
                ]
                if detected_lang in indic_langs and confidence > 0.2:
                    # Boost confidence for Indic languages (Whisper often underestimates)
                    confidence = min(confidence * 1.5, 0.95)

                return (detected_lang, confidence)
            except Exception as e:
                log(f"[Audio] Detection inner error: {e}")
                return ("en", 0.0)

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

            # Cluster IDs now use db.get_next_voice_cluster_id() for uniqueness

            for _idx, seg in enumerate(voice_segments or []):
                audio_path: str | None = None
                global_speaker_id = f"unknown_{uuid.uuid4().hex[:8]}"
                voice_cluster_id = -1

                # Respect explicit SILENCE label
                if seg.speaker_label == "SILENCE":
                    global_speaker_id = "SILENCE"

                # Check Global Registry if embedding exists
                if seg.embedding is not None and global_speaker_id != "SILENCE":
                    match = await self.db.match_speaker(
                        seg.embedding,
                        threshold=settings.voice_clustering_threshold,
                    )
                    if match:
                        global_speaker_id, existing_cluster_id, _score = match
                        voice_cluster_id = existing_cluster_id
                        # If existing matched speaker has no cluster ID (-1), generate one?
                        # Usually it should have one if we follow this new logic consistently.
                        if voice_cluster_id == -1:
                            voice_cluster_id = (
                                self.db.get_next_voice_cluster_id()
                            )
                    else:
                        # New Global Speaker -> New Cluster
                        global_speaker_id = f"SPK_{uuid.uuid4().hex[:12]}"
                        voice_cluster_id = self.db.get_next_voice_cluster_id()

                        self.db.upsert_speaker_embedding(
                            speaker_id=global_speaker_id,
                            embedding=seg.embedding,
                            media_path=str(path),
                            start=seg.start_time,
                            end=seg.end_time,
                            voice_cluster_id=voice_cluster_id,
                        )

                # ALWAYS extract audio clip for every segment (not just those with embeddings)
                audio_extraction_success = False
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
                            "-t",
                            str(seg.end_time - seg.start_time),
                            "-q:a",
                            "2",  # High quality MP3
                            "-map",
                            "a",
                            "-loglevel",
                            "error",
                            str(clip_file),
                        ]
                        result = subprocess.run(
                            cmd, capture_output=True, text=True
                        )
                        if result.returncode != 0:
                            logger.warning(
                                f"[Voice] FFmpeg failed ({result.returncode}): {result.stderr[:100]}"
                            )
                except Exception as e:
                    logger.warning(
                        f"[Voice] FFmpeg failed for {clip_name}: {e}"
                    )

                if clip_file.exists() and clip_file.stat().st_size > 0:
                    audio_path = f"/thumbnails/voices/{clip_name}"
                    audio_extraction_success = True
                else:
                    audio_extraction_success = False

                # === STORE VOICE SEGMENT ===
                # Policy: Strict storage - MUST have audio and not be SILENCE
                if not audio_extraction_success:
                    continue

                if global_speaker_id == "SILENCE":
                    continue
                
                # Speech Emotion Recognition (SER)
                emotion_meta = {}
                try:
                    if not hasattr(self, "_ser_analyzer"):
                        from core.processing.speech_emotion import (
                            SpeechEmotionAnalyzer,
                        )

                        self._ser_analyzer = SpeechEmotionAnalyzer()

                    import librosa

                    # Load the clip we just made (resample to 16k for Wav2Vec2)
                    y, sr = librosa.load(str(clip_file), sr=16000)
                    emotion_res = await self._ser_analyzer.analyze(y, sr)
                    if emotion_res:
                        emotion_meta = {
                            "emotion": emotion_res.get("emotion"),
                            "emotion_conf": emotion_res.get("confidence"),
                        }
                except Exception as e:
                    logger.warning(f"[Voice] SER failed: {e}")

                if seg.embedding is not None and audio_extraction_success:
                    # Ensure voice_cluster_id is always valid (never -1 or 0)
                    if voice_cluster_id <= 0:
                        voice_cluster_id = self.db.get_next_voice_cluster_id()
                        logger.info(
                            f"[Voice] Generated fallback cluster ID {voice_cluster_id} "
                            f"for segment {seg.start_time:.2f}-{seg.end_time:.2f}s"
                        )

                    self.db.insert_voice_segment(
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        speaker_label=global_speaker_id,
                        embedding=seg.embedding.tolist() if hasattr(seg.embedding, "tolist") else seg.embedding,
                        audio_path=audio_path,
                        voice_cluster_id=voice_cluster_id,
                        **emotion_meta,
                    )
                elif seg.embedding is None and audio_extraction_success:
                    # Has audio but no embedding - still useful for playback
                    # Store with placeholder embedding
                    logger.info(
                        f"[Voice] Segment {seg.start_time:.2f}-{seg.end_time:.2f}s "
                        f"has audio but no embedding, storing with placeholder"
                    )
                    # Generate a placeholder cluster ID
                    if voice_cluster_id <= 0:
                        voice_cluster_id = self.db.get_next_voice_cluster_id()

                    # Create a zero embedding placeholder (using small epsilon for safe Cosine distance)
                    placeholder_embedding = [
                        1e-6
                    ] * 256  # WeSpeaker embedding size
                    self.db.insert_voice_segment(
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        speaker_label=global_speaker_id,
                        embedding=placeholder_embedding,
                        audio_path=audio_path,
                        voice_cluster_id=voice_cluster_id,
                    )
                else:
                    # No audio AND no embedding - skip completely
                    logger.warning(
                        f"[Voice] Skipping segment {seg.start_time:.2f}-{seg.end_time:.2f}s: "
                        f"no audio (success={audio_extraction_success}), "
                        f"no embedding (has_emb={seg.embedding is not None})"
                    )
        finally:
            if self.voice:
                self.voice.cleanup()
            del self.voice
            self.voice = None
            self._cleanup_memory()

    @observe("audio_events")
    async def _process_audio_events(
        self, path: Path, job_id: str | None = None
    ) -> None:
        """Detects and indexes discrete audio events (CLAP) using streaming chunks.
        
        Uses FFmpeg to stream audio in chunks instead of loading the entire file
        into memory, preventing OOM on long videos (the 55% stall fix).
        
        Design decisions:
        - 30s chunks: Fits ~3MB RAM at 48kHz stereo
        - 5s overlap: Catches events spanning chunk boundaries
        - Per-chunk progress: User always sees what's processing
        - Immediate cleanup: gc.collect() after each chunk
        """
        logger.info(f"Starting audio event detection for {path.name}")
        
        try:
            from core.processing.audio_events import AudioEventDetector
            import subprocess
            
            # Common audio events to detect
            target_classes = [
                "applause", "laughter", "crying", "screaming", "music", 
                "singing", "speech", "shout", "whisper", "doorbell", 
                "knock", "glass breaking", "car horn", "siren", "gunshot", 
                "explosion", "dog barking", "cat meow", "bird chirp", 
                "rain", "thunder", "wind", "water flowing", "footsteps",
                "silence", "typing", "phone ringing", "alarm"
            ]
            
            detector = AudioEventDetector()
            
            # Get duration via FFprobe (doesn't load file into memory)
            try:
                probe_cmd = [
                    'ffprobe', '-v', 'quiet', '-show_entries', 
                    'format=duration', '-of', 'csv=p=0', str(path)
                ]
                duration = float(subprocess.check_output(probe_cmd).decode().strip())
            except Exception as e:
                logger.warning(f"FFprobe failed, falling back to librosa for duration: {e}")
                import librosa
                duration = librosa.get_duration(path=str(path))
            
            # Streaming parameters
            chunk_seconds = 30  # Process 30s at a time
            overlap_seconds = 5  # 5s overlap to catch events at boundaries
            stride_seconds = chunk_seconds - overlap_seconds  # 25s stride
            sample_rate = 48000  # CLAP expected sample rate
            
            total_chunks = max(1, int(duration / stride_seconds) + 1)
            events_stored = 0
            previous_events = []  # For deduplication
            
            # CLAP processing parameters (within each chunk)
            clap_window = 5.0  # 5s windows for CLAP
            clap_stride = 2.5  # 2.5s stride
            
            for chunk_idx in range(total_chunks):
                chunk_start = chunk_idx * stride_seconds
                chunk_end = min(chunk_start + chunk_seconds, duration)
                
                # Skip if we've gone past the end
                if chunk_start >= duration:
                    break
                
                # === PROGRESS UPDATE ===
                if job_id:
                    progress = 45 + (chunk_idx / total_chunks) * 10  # 45% → 55%
                    progress_tracker.update(
                        job_id, 
                        progress, 
                        stage="audio_events",
                        message=f"Detecting audio events: chunk {chunk_idx+1}/{total_chunks} ({chunk_start:.0f}s-{chunk_end:.0f}s)"
                    )
                
                # Stream only this chunk using FFmpeg (doesn't load full file)
                try:
                    audio_chunk = await self._extract_audio_segment(
                        path, chunk_start, chunk_end, sample_rate
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract audio chunk {chunk_idx}: {e}")
                    continue
                
                if audio_chunk is None or len(audio_chunk) == 0:
                    continue
                
                # Split chunk into CLAP windows
                samples_per_window = int(clap_window * sample_rate)
                stride_samples = int(clap_stride * sample_rate)
                
                clap_chunks = []
                for i in range(0, len(audio_chunk) - samples_per_window + 1, stride_samples):
                    window = audio_chunk[i : i + samples_per_window]
                    window_start = chunk_start + (i / sample_rate)
                    clap_chunks.append((window, window_start))
                
                if not clap_chunks:
                    continue
                
                # Batch process this chunk's windows
                try:
                    chunk_events = await detector.detect_events_batch(
                        clap_chunks, target_classes, sample_rate=sample_rate, 
                        top_k=2, threshold=0.15
                    )
                except Exception as e:
                    logger.warning(f"CLAP detection failed for chunk {chunk_idx}: {e}")
                    continue
                
                # Store events with deduplication
                for (window_audio, window_start), events in zip(clap_chunks, chunk_events):
                    if not events:
                        continue
                        
                    window_end = window_start + clap_window
                    
                    for event in events:
                        # Deduplicate events in overlap region
                        if self._is_duplicate_event(event, window_start, previous_events, overlap_seconds):
                            continue
                        
                        self.db.client.upsert(
                            collection_name=self.db.AUDIO_EVENTS_COLLECTION,
                            points=[
                                models.PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector=[0.0],  # Dummy vector, payload search only
                                    payload={
                                        "media_path": str(path),
                                        "start_time": window_start,
                                        "end_time": window_end,
                                        "event_class": event["event"],
                                        "confidence": event["confidence"],
                                    },
                                )
                            ],
                        )
                        events_stored += 1
                        previous_events.append({
                            "event": event["event"],
                            "start_time": window_start,
                        })
                
                # Cleanup chunk memory immediately
                del audio_chunk
                gc.collect()
            
            logger.info(f"Indexed {events_stored} audio events for {path.name}")
            detector.cleanup()
            
        except Exception as e:
            logger.error(f"Audio event detection failed: {e}")
    
    async def _extract_audio_segment(
        self, path: Path, start: float, end: float, sample_rate: int = 48000
    ) -> np.ndarray | None:
        """Extract a specific audio segment using FFmpeg streaming.
        
        This avoids loading the entire file into memory.
        
        Args:
            path: Path to the media file.
            start: Start time in seconds.
            end: End time in seconds.
            sample_rate: Target sample rate (default 48000 for CLAP).
            
        Returns:
            NumPy array of audio samples, or None on failure.
        """
        import subprocess
        import numpy as np
        
        duration = end - start
        
        try:
            # Use FFmpeg to extract just this segment
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(start),  # Seek to start (before input for speed)
                '-i', str(path),
                '-t', str(duration),  # Duration to extract
                '-vn',  # No video
                '-ac', '1',  # Mono
                '-ar', str(sample_rate),  # Target sample rate
                '-f', 'f32le',  # 32-bit float PCM
                '-'  # Output to stdout
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, timeout=60
            )
            
            if result.returncode != 0:
                logger.warning(f"FFmpeg extraction failed: {result.stderr.decode()[:200]}")
                return None
            
            # Convert bytes to numpy array
            audio = np.frombuffer(result.stdout, dtype=np.float32)
            return audio
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Audio extraction timed out for {start}-{end}s")
            return None
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            return None
    
    def _is_duplicate_event(
        self, event: dict, event_time: float, 
        previous_events: list, overlap_window: float
    ) -> bool:
        """Check if an event is a duplicate from the overlap region.
        
        Args:
            event: The event dict with 'event' key.
            event_time: Start time of the event.
            previous_events: List of previously stored events.
            overlap_window: Size of overlap region in seconds.
            
        Returns:
            True if this is a duplicate, False otherwise.
        """
        for prev in previous_events:
            # Same event class within overlap window
            if (prev["event"] == event["event"] and 
                abs(prev["start_time"] - event_time) < overlap_window):
                return True
        return False

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
        from core.processing.extractor import FrameExtractor

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

        # Use persistent cache context to prevent premature deletion
        # This fixes race conditions where frames are deleted before batch processing ends
        with FrameExtractor.FrameCache() as frame_cache_dir:
            # Pass time range to extractor for partial processing
            # NOTE: Extractor now yields ExtractedFrame objects with ACTUAL PTS timestamps
            frame_generator = self.extractor.extract(
                path,
                interval=self.frame_interval_seconds,
                start_time=getattr(self, "_start_time", None),
                end_time=getattr(self, "_end_time", None),
                output_dir=frame_cache_dir,
            )

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

            # Batch Processing Helper (Optimized for SPEED)
            async def _process_batch_items(frames_to_process: list):
                if not frames_to_process:
                    return

                paths = [f.path for f in frames_to_process]
                
                # === 1. BATCH FACE DETECTION ===
                try:
                    batch_faces = await self.faces.detect_faces_batch(paths)
                except Exception as e:
                    logger.warning(f"Batch face detection failed: {e}")
                    batch_faces = [None] * len(frames_to_process)

                # === 2. BATCH VISUAL ENCODING (NEW - Major Speedup) ===
                batch_embeddings = [None] * len(frames_to_process)
                try:
                    if self.vision and hasattr(self.vision, 'encode_batch'):
                        # Use batch encoding for 2-3x speedup
                        embeddings = await self.vision.encode_batch(paths)
                        batch_embeddings = embeddings
                        logger.debug(f"[Vision] Batch encoded {len(paths)} frames")
                    elif self.vision:
                        # Fallback to sequential if no batch method
                        for i, p in enumerate(paths):
                            try:
                                batch_embeddings[i] = await self.vision.encode_image(p)
                            except Exception:
                                batch_embeddings[i] = None
                except Exception as e:
                    logger.warning(f"Batch visual encoding failed: {e}")
                    batch_embeddings = [None] * len(frames_to_process)

                # === 3. PROCESS EACH FRAME WITH PRE-COMPUTED DATA ===
                for idx, frame_item in enumerate(frames_to_process):
                    f_path = frame_item.path
                    f_ts = frame_item.timestamp
                    f_idx = frame_item.frame_index
                    
                    # Update context
                    narrative_context = temporal_ctx.get_context_for_vlm()
                    neighbor_timestamps = [c.timestamp for c in temporal_ctx.sensory]

                    new_desc = await self._process_single_frame(
                        video_path=path,
                        frame_path=f_path,
                        timestamp=f_ts,
                        index=f_idx,
                        context=narrative_context,
                        neighbor_timestamps=neighbor_timestamps,
                        pre_detected_faces=batch_faces[idx] if idx < len(batch_faces) else None,
                        pre_computed_embedding=batch_embeddings[idx] if idx < len(batch_embeddings) else None,
                    )

                    if new_desc:
                        # Add to temporal context memory
                        t_ctx = TemporalContext(
                            timestamp=f_ts,
                            description=new_desc[:200],
                            faces=list(self._face_clusters.keys())
                            if hasattr(self, "_face_clusters")
                            else [],
                        )
                        temporal_ctx.add_frame(t_ctx)

                        # Add to Scenelet Builder
                        s_ctx = TemporalContext(
                            timestamp=f_ts,
                            description=new_desc,
                            faces=list(self._face_clusters.keys())
                            if hasattr(self, "_face_clusters")
                            else [],
                        )
                        scenelet_builder.add_frame(s_ctx)
                    
                    # Cleanup processed frame immediately
                    if f_path.exists():
                         try:
                             f_path.unlink()
                         except Exception:
                             pass

            pending_frames = []


            async for extracted_frame in frame_generator:
                if job_id:
                    if progress_tracker.is_cancelled(
                        job_id
                    ) or progress_tracker.is_paused(job_id):
                        break

                # ExtractedFrame contains: path, timestamp (actual PTS), frame_index
                frame_path = extracted_frame.path
                frame_count = extracted_frame.frame_index
                
                # RESUME: Skip already processed frames
                if frame_count < resume_from_frame:
                    if frame_path.exists():
                        frame_path.unlink()
                    continue

                # USE ACTUAL PTS TIMESTAMP (from FFprobe, not calculated)
                # This fixes timestamp drift on VFR videos
                timestamp = extracted_frame.timestamp
                
                if self.frame_sampler.should_sample(frame_count):
                    pending_frames.append(extracted_frame)
                    if len(pending_frames) >= 16:
                        await _process_batch_items(pending_frames)
                        pending_frames = []

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
                            probe_data = await self.prober.probe(path)
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
                # Only delete if NOT in pending batch (processed frames are deleted by helper)
                if extracted_frame not in pending_frames:
                    if frame_path.exists():
                        try:
                            frame_path.unlink()
                        except Exception:
                            pass

                # frame_count is already set from extracted_frame.frame_index

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

            # Process any remaining frames in the batch
            if pending_frames:
                await _process_batch_items(pending_frames)
                pending_frames = []

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
                    await self.db.store_scenelet(
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
        scenes = await detect_scenes(path)
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

            # 6. Deep Research Analysis (Cinematography, Aesthetics, Mood)
            dr_meta = {}
            internvideo_features = None
            languagebind_features = None
            
            if frame_bytes:
                try:
                    import cv2
                    import numpy as np

                    from core.processing.deep_research import (
                        get_deep_research_processor,
                    )

                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    processor = get_deep_research_processor()
                    # Run deep research on the representative frame
                    dr_result = await processor.analyze_frame(
                        img_np,
                        compute_embeddings=False, # VectorDB does this
                        compute_saliency=False,
                        compute_fingerprint=True
                    )
                    
                    dr_meta = {
                        "shot_type": dr_result.shot_type,
                        "mood": dr_result.mood,
                        "aesthetic_score": dr_result.aesthetic_score,
                        "is_black_frame": dr_result.is_black_frame,
                    }
                    
                    # Video understanding embeddings (InternVideo, LanguageBind)
                    # Only compute if enabled in settings (saves VRAM on low-end systems)
                    if settings.enable_video_embeddings:
                        try:
                            video_result = await processor.analyze_video_segment(
                                video_path=path,
                                start_time=scene.start_time,
                                end_time=scene.end_time,
                                sample_frames=8,
                            )
                            
                            # Extract video embeddings for action search
                            if "internvideo" in video_result.video_features:
                                internvideo_features = video_result.video_features["internvideo"].tolist()
                            if "languagebind" in video_result.video_features:
                                languagebind_features = video_result.video_features["languagebind"].tolist()
                                
                            logger.debug(f"Scene {idx}: InternVideo={internvideo_features is not None}, LanguageBind={languagebind_features is not None}")
                        except Exception as e:
                            logger.debug(f"Video understanding failed for scene {idx}: {e}")
                            
                except Exception as e:
                    logger.warning(f"Deep Research failed for scene {idx}: {e}")

            # 7. Generate CLIP/SigLIP visual features for true multimodal search
            visual_features = None
            if settings.enable_visual_embeddings and frame_bytes:
                try:
                    # Lazy load the encoder to save VRAM until needed
                    if self._visual_encoder is None:
                        from core.processing.visual_encoder import get_default_visual_encoder
                        logger.info("Initializing Visual Encoder (CLIP/SigLIP) for ingestion...")
                        self._visual_encoder = get_default_visual_encoder()
                    
                    # Convert frame bytes to numpy array
                    import io
                    from PIL import Image
                    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                    img_np = np.array(img)
                    
                    # Encode the frame to get visual embeddings
                    visual_features_arr = await self._visual_encoder.encode_image(img_np)
                    if visual_features_arr is not None:
                        visual_features = visual_features_arr.tolist() if hasattr(visual_features_arr, 'tolist') else list(visual_features_arr)
                        logger.debug(f"Scene {idx}: Visual features generated (dim={len(visual_features)})")
                except Exception as e:
                    logger.warning(f"Failed to generate visual features for scene {idx}: {e}")
                    visual_features = None

            # 8. Store scene with multi-vector
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
                    # Deep Research Metadata
                    "shot_type": dr_meta.get("shot_type", ""),
                    "mood": dr_meta.get("mood", ""),
                    "aesthetic_score": dr_meta.get("aesthetic_score", 0.0),
                    # Non-indexed data
                    "visual_summary": visual_summary,
                    "action_sequence": aggregated.get("action_sequence", ""),
                    "location": aggregated.get("location", ""),
                    "cultural_context": aggregated.get("cultural_context", ""),
                    "dialogue_transcript": dialogue_text,
                    "frame_count": aggregated.get("frame_count", 0),
                    "thumbnail_path": thumb_path,
                }

                # Enhance visual text with Deep Research insights
                if dr_meta:
                     visual_text += (
                         f" {dr_meta.get('shot_type', '')} "
                         f"{dr_meta.get('mood', '')} "
                         f"aesthetic_score: {dr_meta.get('aesthetic_score', 0):.2f}"
                     )

                await self.db.store_scene(
                    media_path=str(path),
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    visual_text=visual_text,
                    motion_text=motion_text,
                    dialogue_text=dialogue_text,
                    visual_features=visual_features,  # CLIP/SigLIP embedding
                    internvideo_features=internvideo_features,
                    languagebind_features=languagebind_features,
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
        pre_detected_faces: list | None = None,
        pre_computed_embedding: list[float] | None = None,
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
            pre_detected_faces: Optional list of faces detected in batch mode.
            pre_computed_embedding: Optional visual embedding from batch encoding.

        Returns:
            The generated frame description or None if processing failed.
        """
        if not self.vision or not self.faces:
            return None

        # 1. DETECT FACES FIRST (Capture Identity before Vision)
        face_cluster_ids: list[int] = []
        detected_faces = []
        try:
            if pre_detected_faces is not None:
                detected_faces = pre_detected_faces
            else:
                detected_faces = await self.faces.detect_faces(frame_path)
        except Exception:
            pass

        if hasattr(self, "_face_track_builder") and detected_faces:
            self._face_track_builder.process_frame(
                faces=detected_faces,
                frame_index=index,
                timestamp=timestamp,
            )

        # ------------------------------------------------------------
        # DEEP RESEARCH: SOTA Frame Analysis (Cinematography, Aesthetics)
        # OPTIMIZATION: Skip per-frame if deep_research_per_scene is True
        # (Will run on scene keyframes instead via _process_scene_captions)
        # ------------------------------------------------------------
        dr_result = None
        # Check global master switch first
        if getattr(settings, 'enable_deep_research', True):
            skip_deep_research = getattr(settings, 'deep_research_per_scene', True)
            if not skip_deep_research:
                try:
                    dr_processor = get_deep_research_processor()
                    # Run analysis (fire and forget features for now, use metadata)
                    dr_result = await dr_processor.analyze_frame(
                        frame=frame_path,
                        compute_aesthetics=True,
                        compute_saliency=False,  # Skip heavy saliency for speed
                        compute_fingerprint=True,
                    )
                    if dr_result:
                        logger.info(
                            f"[DeepResearch] Frame {timestamp:.2f}s: "
                            f"Shot='{dr_result.shot_type}', Mood='{dr_result.mood}', "
                            f"Aesthetic={dr_result.aesthetic_score:.2f}"
                        )
                except Exception as e:
                    logger.warning(f"[DeepResearch] Analysis failed: {e}")
        
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
                # Use a neutral placeholder to avoid confusing the VLM with "Unknown (cluster X)"
                # which leads to descriptions like "Unknown (cluster 15) is walking..."
                identity_parts.append(f"Person {idx + 1}")

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
            # OPTIMIZATION: Skip OCR on visually similar frames (perceptual hash)
            # ============================================================
            ocr_text = ""
            ocr_boxes = []
            try:
                # Load frame as numpy array (Windows path safe)
                import cv2
                import numpy as np

                frame_data = np.fromfile(str(frame_path), dtype=np.uint8)
                frame_img = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

                if frame_img is not None:
                    # --- OCR Skip-Unchanged-Frames Optimization ---
                    skip_ocr = False
                    ocr_skip_enabled = getattr(settings, 'ocr_skip_unchanged_frames', True)
                    
                    if ocr_skip_enabled:
                        try:
                            # Compute perceptual hash (fast, 8x8 grayscale downsample)
                            gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
                            resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
                            mean_val = np.mean(resized)
                            current_hash = (resized > mean_val).flatten().tobytes()
                            
                            # Compare with previous frame's hash
                            if hasattr(self, '_last_ocr_hash') and self._last_ocr_hash is not None:
                                # Hamming distance (count of differing bits)
                                diff = sum(a != b for a, b in zip(current_hash, self._last_ocr_hash))
                                if diff < 8:  # <12.5% difference = same frame
                                    skip_ocr = True
                                    if hasattr(self, '_last_ocr_text'):
                                        ocr_text = self._last_ocr_text  # Reuse previous result
                                        ocr_boxes = getattr(self, '_last_ocr_boxes', [])
                                    logger.debug(f"[OCR] Skipped unchanged frame (hash diff: {diff})")
                            
                            self._last_ocr_hash = current_hash
                        except Exception as e:
                            logger.debug(f"[OCR] Hash comparison failed: {e}")
                    
                    # Gate: Only run OCR if frame likely contains text (edge density check)
                    # Check enable_ocr master switch
                    if getattr(settings, 'enable_ocr', True):
                        if not skip_ocr and self.text_gate.has_text(frame_img):
                            ocr_result = await self.ocr_engine.extract_text(
                                frame_img
                            )
                            if ocr_result and ocr_result.get("text"):
                                ocr_text = ocr_result["text"]
                                ocr_boxes = ocr_result.get("boxes", [])
                                # Cache for skip-unchanged optimization
                                self._last_ocr_text = ocr_text
                                self._last_ocr_boxes = ocr_boxes
                                logger.info(f"[OCR] Extracted: {ocr_text[:100]}...")
                            else:
                                logger.debug("[OCR] No text found in gated frame")
                    else:
                        logger.debug("[OCR] Disabled via config")

                    # Deep Research Enrichment (Safety & Time)
                    try:
                        # Content Moderation
                        if getattr(settings, 'enable_content_moderation', False):
                            if not hasattr(self, "_moderator"):
                                from core.processing.content_moderation import (
                                    VisualContentModerator,
                                )
                                self._moderator = VisualContentModerator()

                            safe_res = await self._moderator.check_frame(frame_img)
                            if not safe_res.is_safe:
                                flags_str = ", ".join(
                                    [f.name for f in safe_res.flags]
                                )
                                video_context += f"\n[SAFETY-FLAG]: {flags_str}"
                        
                        # Clock Reader
                        if getattr(settings, 'enable_time_extraction', False):
                            if not hasattr(self, "_clock"):
                                from core.processing.clock_reader import ClockReader
                                self._clock = ClockReader()

                            clock_res = await self._clock.read(frame_img)
                            if clock_res:
                                video_context += (
                                    f"\n[VISIBLE-TIME]: {clock_res.get('time_string')}"
                                )
                    except Exception as e:
                        logger.warning(f"Deep Research enrichment error: {e}")

            except Exception as e:
                logger.warning(f"[OCR] Failed: {e}")

            # ============================================================
            # OBJECT DETECTION: YOLO-World for general objects (lazy-loaded)
            # ============================================================
            detected_objects: list[str] = []
            try:
                if self.enhanced_config and self.enhanced_config.object_detector:
                    obj_detector = self.enhanced_config.object_detector
                    if frame_img is not None:
                        # Run YOLO-World detection on frame
                        detections = obj_detector.detect(frame_img)
                        if detections:
                            detected_objects = [d.get("label", d.get("class", "")) for d in detections if d.get("confidence", 0) > 0.3]
                            if detected_objects:
                                unique_objects = list(set(detected_objects))
                                video_context += f"\n[DETECTED-OBJECTS]: {', '.join(unique_objects)}"
                                logger.debug(f"[ObjectDetection] Found: {unique_objects}")
            except Exception as e:
                logger.debug(f"[ObjectDetection] Skipped: {e}")

            # Run structured vision analysis
            async with VLM_SEMAPHORE:
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
                async with VLM_SEMAPHORE:
                    description = await self.vision.describe(
                        frame_path, context=context
                    )
            except Exception:
                pass

        if description:
            vector = (await self.db.encode_texts(description))[0]

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
                # Merge YOLO-World detected objects into entities
                if detected_objects:
                    vlm_entities = set(e.lower() for e in payload["entities"])
                    for obj in detected_objects:
                        if obj.lower() not in vlm_entities:
                            payload["entities"].append(obj)
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
                
                # DYNAMIC: Extract ALL visual attributes from entities for searchability
                # NO HARDCODING - works for any entity type (clothing, vehicles, objects, etc.)
                # This enables queries like "light green shirt", "red ferrari", "blue bag"
                visual_attributes = []
                entity_details = []
                
                # Extract from ALL entities - let VLM determine what's important
                for entity in analysis.entities:
                    # Collect ALL visual details (colors, patterns, textures, states)
                    if entity.visual_details:
                        visual_attributes.append(entity.visual_details.lower())
                    
                    # Collect entity names for keyword search
                    entity_details.append(entity.name.lower())
                    
                    # Also collect category for filtering
                    if entity.category:
                        entity_details.append(entity.category.lower())
                
                # Store as searchable fields - hybrid search will match these
                if visual_attributes:
                    payload["visual_attributes"] = visual_attributes
                if entity_details:
                    payload["entity_details"] = entity_details

            # DEEP RESEARCH METADATA INJECTION
            if dr_result:
                payload["cinematography"] = {
                    "shot_type": dr_result.shot_type,
                    "shot_confidence": dr_result.shot_confidence,
                    "mood": dr_result.mood,
                    "mood_confidence": dr_result.mood_confidence,
                    "aesthetic_score": dr_result.aesthetic_score,
                    "is_black_frame": dr_result.is_black_frame,
                    "blur_score": dr_result.blur_score,
                    "perceptual_hash": dr_result.perceptual_hash,
                }
                # Enrich search text with high-confidence tags
                if dr_result.shot_confidence > 0.4:
                    description += f". Shot type: {dr_result.shot_type}"
                if dr_result.mood_confidence > 0.4:
                    description += f". Mood: {dr_result.mood}"

            # 3d. Add structured face data for UI overlays (bboxes)
            # This enables drawing boxes around identified people in the UI
            faces_metadata = []
            for face, cluster_id in zip(
                detected_faces, face_cluster_ids, strict=False
            ):
                face_name = self.db.get_face_name_by_cluster(cluster_id)
                bbox = face.bbox if isinstance(face.bbox, list) else list(face.bbox)
                # Calculate bbox dimensions for analytics
                top, right, bottom, left = bbox
                bbox_width = right - left
                bbox_height = bottom - top
                bbox_area = bbox_width * bbox_height
                
                faces_metadata.append(
                    {
                        "bbox": bbox,  # [top, right, bottom, left]
                        "cluster_id": cluster_id,
                        "name": face_name,
                        "confidence": face.confidence,
                        # NEW: Detection timestamp and dimensions for enriched analytics
                        "detected_at": timestamp,
                        "bbox_width": bbox_width,
                        "bbox_height": bbox_height,
                        "bbox_area": bbox_area,
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

            # 3f. Add pre-computed visual embedding (from batch processing)
            if pre_computed_embedding is not None:
                payload["visual_embedding"] = pre_computed_embedding

            # 3f. Generate Vector (Include identity for searchability)
            # Avoid duplicating names if vision model already detected them
            full_text = description
            if "identity_text" in payload:
                # Only add identity_text if names aren't already in description
                identity_names = payload.get("face_names", []) + payload.get(
                    "speaker_names", []
                )
                names_not_in_desc = [
                    name
                    for name in identity_names
                    if name and name.lower() not in description.lower()
                ]
                if names_not_in_desc:
                    # Build identity suffix only for missing names
                    identity_suffix = ""
                    if names_not_in_desc:
                        identity_suffix = (
                            f"Visible: {', '.join(names_not_in_desc)}"
                        )
                    full_text = (
                        f"{description}. {identity_suffix}"
                        if identity_suffix
                        else description
                    )

            vector = (await self.db.encode_texts(full_text))[0]

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

    @observe("post_processing")
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

            # ------------------------------------------------------------
            # SOTA SCENE INDEXING (Adaptive Scene Segmentation)
            # ------------------------------------------------------------
            try:
                from core.processing.scene_aggregator import (
                    aggregate_frames_to_scene,
                )
                from core.processing.scene_detector import detect_scenes

                # 1. Detect logical scenes (shots)
                raw_scenes = detect_scenes(media_path)
                if not raw_scenes:
                    # Fallback: Treat whole video as one scene if detection fails or single shot
                    raw_scenes = []  # Will be handled by global context or we can make 1 synthetic scene

                logger.info(
                    f"Indexing {len(raw_scenes)} scenes for {path.name}"
                )

                # 2. Process each scene
                for scene_idx, scene_info in enumerate(raw_scenes):
                    # Find frames within this scene's time window
                    scene_frames = [
                        f
                        for f in frames
                        if scene_info.start_time
                        <= f.get("timestamp", 0)
                        < scene_info.end_time
                    ]

                    if not scene_frames:
                        # Skip scenes with no analyzed frames (avoid empty noise)
                        continue

                    # Aggregate frame data into scene-level metadata
                    scene_data = aggregate_frames_to_scene(
                        frames=scene_frames,
                        start_time=scene_info.start_time,
                        end_time=scene_info.end_time,
                        dialogue_segments=audio_segments,
                    )

                    # 3. Generate Multi-Vector Embeddings (Visual, Motion, Dialogue, Hybrid)
                    # Visual: Summary of actions and entities
                    visual_text = scene_data.get("visual_summary", "")

                    # Motion: Sequence of actions
                    motion_text = scene_data.get("action_sequence", "")

                    # Dialogue: Transcript
                    dialogue_text = scene_data.get("dialogue_transcript", "")

                    # Hybrid: Combined rich description for general search
                    hybrid_text = (
                        f"{visual_text}. {motion_text}. "
                        f"{dialogue_text}. "
                        f"Location: {scene_data.get('location')}."
                    )

                    # Encode vectors
                    # Note: We use the same encoder for all modalities (shared latent space) or specific ones if available
                    # For now, using the main encoder for all text representations is standard for dense retrieval
                    vectors = {}
                    if visual_text:
                        vectors["visual"] = await self.db.encode_text(visual_text)
                    if motion_text:
                        vectors["motion"] = await self.db.encode_text(motion_text)
                    if dialogue_text:
                        vectors["dialogue"] = await self.db.encode_text(dialogue_text)

                    # Generate deterministic ID
                    import uuid

                    scene_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{media_path}_scene_{scene_idx}",
                        )
                    )

                    # Upsert to SCENES collection
                    self.db.client.upsert(
                        collection_name=self.db.SCENES_COLLECTION,
                        points=[
                            models.PointStruct(
                                id=scene_id,
                                vector=vectors,  # Multi-vector dictionary
                                payload={
                                    "media_path": media_path,
                                    "video_path": media_path,  # Alias
                                    "start_time": scene_info.start_time,
                                    "end_time": scene_info.end_time,
                                    "scene_index": scene_idx,
                                    "description": hybrid_text,  # Main display text
                                    **scene_data,  # flattened metadata
                                },
                            )
                        ],
                    )

                # Add simplified global scene for context
                # (Existing logic below can remain or be replaced by this granular loop metadata)

            except Exception as e:
                logger.error(f"Scene indexing failed: {e}")

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

                scene_data_global = {
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
                global_ctx.add_scene(scene_data_global)
                
                # Wire to Knowledge Graph / GraphRAG for social network and timeline queries
                try:
                    from core.storage.identity_graph import identity_graph
                    
                    # Get face_cluster_ids from all frames
                    all_face_cluster_ids = list({
                        cid for f in frames
                        for cid in f.get("face_cluster_ids", [])
                    })
                    
                    # Create scene in SQLite graph database
                    identity_graph.create_scene(
                        media_id=media_path,
                        start_time=scene_data_global["start_time"],
                        end_time=scene_data_global["end_time"],
                        location=scene_data_global.get("location"),
                        description=scene_data_global.get("visual_summary"),
                        face_cluster_ids=all_face_cluster_ids,
                        entities=scene_data_global.get("entities", []),
                        actions=[f.get("action", "") for f in frames[:10] if f.get("action")],
                    )
                    logger.debug(f"[GraphRAG] Created scene with {len(all_face_cluster_ids)} faces")
                except Exception as e:
                    logger.debug(f"[GraphRAG] Scene creation skipped: {e}")

                global_summary = global_ctx.to_payload()

                # Generate and attach Main Video Thumbnail
                try:
                    main_thumb = await self._generate_main_thumbnail(path)
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

    async def _generate_main_thumbnail(self, path: Path) -> str | None:
        """Generates a representative thumbnail for the video at 5.0s.

        Args:
            path: Path to the media file.

        Returns:
            The relative web path to the generated thumbnail, or None on failure.
        """
        import hashlib

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

            # Extract at 5 seconds using async subprocess (non-blocking)
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

            # Run ffmpeg asynchronously (doesn't block event loop)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

            # Fallback to 0s if 5s failed (e.g. short video)
            if not thumb_file.exists():
                cmd[3] = "00:00:00.000"
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()

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
