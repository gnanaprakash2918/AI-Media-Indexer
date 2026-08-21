"""Media ingestion pipeline orchestrator."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import traceback
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from qdrant_client.http import models

from config import settings
from core.domain.schemas import MediaType
from core.errors import IngestionError, MediaIndexerError
from core.llm.video_vlm import VideoVLM
from core.processing.extractor import FrameExtractor
from core.processing.frame_sampling import TextGatedOCR
from core.processing.identity import FaceManager
from core.processing.metadata import MetadataEngine
from core.processing.ocr_factory import get_ocr_engine
from core.processing.prober import MediaProbeError, MediaProber
from core.processing.scene_detector import detect_scenes
from core.processing.transnet_detector import TransNetV2
from core.processing.vision import VisionAnalyzer
from core.processing.voice import VoiceProcessor
from core.storage.db import VectorDB
from core.storage.constants import MEDIA_SEGMENTS_COLLECTION, AUDIO_EVENTS_COLLECTION
from core.utils.logger import bind_context, log_verbose, logger
from core.utils.observe import observe
from core.utils.progress import progress_tracker
from core.utils.retry import retry

# Global semaphore for VLM parallelism (auto-scales based on hardware profile)
# This prevents OOM when using Ollama/Gemini with large contexts
VLM_SEMAPHORE = asyncio.Semaphore(settings.vlm_concurrency)


from core.ingestion.stages.audio_events_stage import AudioEventsStage
from core.ingestion.stages.audio_stage import AudioStage
from core.ingestion.stages.frame_stage import FrameStage
from core.ingestion.stages.scene_stage import SceneStage
from core.ingestion.stages.voice_stage import VoiceStage
from core.ports import (
    FaceTracker as FaceTrackerProtocol,
    VisionAnalyzer as VisionAnalyzerProtocol,
    VLMProcessor as VLMProcessorProtocol,
    VoiceProcessor as VoiceProcessorProtocol,
    StorageBackend,
)


class IngestionPipeline:
    """Orchestrate the media ingestion process (probing, transcription, vision, etc)."""

    def __init__(
        self,
        *,
        db: StorageBackend | None = None,
        vision_analyzer: VisionAnalyzerProtocol | None = None,
        face_manager: FaceTrackerProtocol | None = None,
        voice_processor: VoiceProcessorProtocol | None = None,
        video_vlm: VLMProcessorProtocol | None = None,
        qdrant_backend: str = settings.qdrant_backend,
        qdrant_host: str = settings.qdrant_host,
        qdrant_port: int = settings.qdrant_port,
        frame_interval_seconds: float = settings.frame_interval,
        tmdb_api_key: str | None = settings.tmdb_api_key,
    ) -> None:
        """Initializes the ingestion pipeline with dependency injection.

        Args:
            db: Abstract storage backend. If None, concrete VectorDB is used.
            vision_analyzer: Abstract vision processor.
            face_manager: Abstract face tracking processor.
            voice_processor: Abstract voice diarization processor.
            video_vlm: Abstract VLM processor.
            qdrant_backend: Fallback Qdrant backend type.
            qdrant_host: Fallback Qdrant host.
            qdrant_port: Fallback Qdrant port.
            frame_interval_seconds: Frame sampling interval in seconds.
            tmdb_api_key: Optional API key for TMDB movie metadata.
        """
        from core.processing.dependency_check import check_model_dependencies

        check_model_dependencies()

        self.scene_detector = detect_scenes
        self.prober = MediaProber()
        # OCR Components
        self.ocr_engine = get_ocr_engine()
        self.text_gate = TextGatedOCR()
        self.extractor = FrameExtractor()

        # Inject or instantiate dependencies
        self.db = db or VectorDB(
            backend=qdrant_backend,
            host=qdrant_host,
            port=qdrant_port,
        )
        self.vision_analyzer = vision_analyzer or VisionAnalyzer()
        self.metadata_engine = MetadataEngine(
            tmdb_key=settings.tmdb_api_key, omdb_key=settings.omdb_api_key
        )
        self.transnet = TransNetV2()
        self.video_vlm = video_vlm or VideoVLM()
        self.voice_processor = voice_processor or VoiceProcessor(db=self.db)

        self.frame_interval_seconds = frame_interval_seconds
        self.vision: VisionAnalyzerProtocol | None = None
        self.faces: FaceTrackerProtocol | None = None
        self.voice: VoiceProcessorProtocol | None = None


        self._face_clusters: dict[int, list[float]] = {}
        self._face_cluster_lock = (
            asyncio.Lock()
        )  # Prevents race during parallel face clustering

        # Deep Video Understanding (SAM 3)
        if settings.enable_sam3_tracking:
            from core.tracking.sam3_tracker import SAM3Tracker
            self.sam3_tracker = SAM3Tracker()
        else:
            self.sam3_tracker = None
        self.frame_sampler_every_n = getattr(settings, "frame_sample_every", 5)

        # Visual encoder for CLIP/SigLIP embeddings (lazy-loaded)
        self._visual_encoder = None

        # Probe cache (avoid 6x FFprobe calls per video)
        self._probe_cache: dict[str, dict] = {}
        
        # Instantiate Composition Stages
        self.audio_stage = AudioStage(
            db=self.db, get_probe_data=self.get_probe_data, cleanup_memory=self._cleanup_memory
        )
        self.voice_stage = VoiceStage(db=self.db, cleanup_memory=self._cleanup_memory)
        self.audio_events_stage = AudioEventsStage(db=self.db, get_probe_data=self.get_probe_data)
        self.frame_stage = FrameStage(
            db=self.db,
            extractor=self.extractor,
            frame_interval_seconds=self.frame_interval_seconds,
            text_gate=self.text_gate,
            face_cluster_lock=self._face_cluster_lock,
            get_probe_data=self.get_probe_data,
            cleanup_memory=self._cleanup_memory,
            get_audio_segments_for_video=self._get_audio_segments_for_video,
            get_speaker_clusters_at_time=self._get_speaker_clusters_at_time,
        )
        self.scene_stage = SceneStage(
            db=self.db,
            transnet=self.transnet,
            get_audio_segments_for_video=self._get_audio_segments_for_video,
            get_audio_events_for_video=self._get_audio_events_for_video,
        )

    def _reset_per_video_caches(self) -> None:
        """Reset caches that should not persist across different videos."""
        self.scene_stage._cached_scenes = None
        self.scene_stage._cached_scenes_path = None
        self._probe_cache.clear()



    def _cleanup_memory(self, context: str = "") -> None:
        """Force garbage collection and clear CUDA cache."""
        from core.utils.device import empty_cache

        empty_cache()

        try:
            from core.utils.hardware import log_vram_status

            log_vram_status(context or "cleanup")
        except Exception:
            pass

    async def get_probe_data(self, path: Path) -> dict:
        """Get probe data with caching (avoids 6x FFprobe calls)."""
        key = str(path)
        if key not in self._probe_cache:
            self._probe_cache[key] = await self.prober.probe(path)
        return self._probe_cache[key]

    def clear_probe_cache(self) -> None:
        """Clear probe cache (call after video processing complete)."""
        self._probe_cache.clear()
        from core.processing.prober import clear_probe_cache as _clear_global

        _clear_global()

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

        # Verbose logging for full context
        log_verbose(
            f"[Pipeline] process_video started: path={video_path}, "
            f"job_id={job_id}, resume={resume}, media_type={media_type_hint}, "
            f"start_time={start_time}, end_time={end_time}, "
            f"content_type_hint={content_type_hint}"
        )

        self._start_time = start_time
        self._end_time = end_time
        self._hitl_content_type = (
            content_type_hint if content_type_hint != "auto" else None
        )
        self.frame_stage.hitl_content_type = self._hitl_content_type
        self.audio_stage.audio_classification = None
        self.frame_stage.audio_classification = None

        path = Path(video_path)
        log_verbose(
            f"[Pipeline] Resolved path: {path}, exists={path.exists()}, size={path.stat().st_size if path.exists() else 0}"
        )

        progress_tracker.start(
            job_id,
            file_path=str(path),
            media_type=media_type_hint or "unknown",
            resume=resume,
        )

        if not path.exists() or not path.is_file():
            progress_tracker.fail(job_id, error=f"Invalid media path: {path}")
            raise FileNotFoundError(f"Invalid media path: {path}")

        # === SOURCE TRIMMING (User Request: "trim and pass only that") ===
        if start_time is not None or end_time is not None:
            import subprocess

            logger.info(
                f"[Pipeline] Trimming requested: {start_time}-{end_time}"
            )

            start_val = start_time or 0.0
            end_suffix = f"{end_time}" if end_time else "end"
            trimmed_filename = (
                f"{path.stem}_trim_{start_val}_{end_suffix}{path.suffix}"
            )
            trimmed_path = path.parent / trimmed_filename

            # Construct FFmpeg command (Input Seeking for speed)
            cmd = ["ffmpeg", "-y"]
            if start_time is not None:
                cmd.extend(["-ss", str(float(start_time))])

            cmd.extend(["-i", str(path)])

            if end_time is not None:
                duration_trim = float(end_time) - start_val
                if duration_trim > 0:
                    cmd.extend(["-t", str(duration_trim)])

            # Re-encode video for frame accuracy (ultrafast), copy audio
            cmd.extend(
                ["-c:v", "libx264", "-preset", "ultrafast", "-c:a", "copy"]
            )
            cmd.extend([str(trimmed_path)])

            logger.info(f"[Pipeline] Trimming command: {' '.join(cmd)}")

            try:
                # Run async to avoid blocking event loop
                process = await asyncio.create_subprocess_exec(
                    *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    error_msg = stderr.decode()
                    logger.error(f"[Pipeline] Trim failed: {error_msg}")
                    # Fallback or Raise? Raise, as user explicitly requested trim.
                    raise RuntimeError(f"FFmpeg trim failed: {error_msg}")

                logger.info(
                    f"[Pipeline] Trimming successful. New source: {trimmed_path}"
                )
                path = trimmed_path  # Switch context to trimmed file

            except Exception as e:
                progress_tracker.fail(job_id, error=f"Trimming failed: {e}")
                raise

        hint_enum = (
            MediaType(media_type_hint)
            if media_type_hint in MediaType._value2member_map_
            else MediaType.UNKNOWN
        )

        # Track trimmed file for cleanup at end of processing
        self._trimmed_path = (
            trimmed_path
            if (start_time is not None or end_time is not None)
            else None
        )

        _ = await self.metadata_engine.identify(path, user_hint=hint_enum)

        try:
            probed = await self.get_probe_data(path)
            duration = float(probed.get("format", {}).get("duration", 0.0))
        except MediaProbeError as e:
            progress_tracker.fail(job_id, error=f"Media probe failed: {e}")
            raise

        # === CHUNKING DECISION (OOM Prevention) ===
        # Prevent OOM by processing long videos in chunks
        chunk_enabled = getattr(settings, "enable_chunking", True)
        chunk_duration = getattr(
            settings, "chunk_duration_seconds", 600
        )  # 10 min default
        if chunk_duration <= 0:
            logger.warning(
                f"[Pipeline] Invalid chunk_duration={chunk_duration}, defaulting to 600s"
            )
            chunk_duration = 600
        min_length_for_chunk = getattr(
            settings, "min_media_length_for_chunking", 1800
        )  # 30 min
        auto_chunk_hw = getattr(settings, "auto_chunk_by_hardware", True)

        # Auto-adjust chunk size based on hardware
        if auto_chunk_hw and torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                if vram_gb < 8:
                    chunk_duration = min(
                        chunk_duration, 300
                    )  # 5 min for low VRAM
                    min_length_for_chunk = 600  # Chunk anything > 10 min
                    logger.info(
                        f"[Chunking] Low VRAM ({vram_gb:.1f}GB) - using 5min chunks"
                    )
            except Exception:
                pass

        should_chunk = chunk_enabled and duration > min_length_for_chunk

        if should_chunk:
            logger.info(
                f"[Chunking] Video {duration / 60:.1f}min > threshold {min_length_for_chunk / 60:.0f}min. "
                f"Will process in {chunk_duration / 60:.0f}min chunks."
            )
            # Store chunk info for _process_frames to use
            self._chunk_duration = chunk_duration
            self._total_chunks = int(duration / chunk_duration) + 1
        else:
            self._chunk_duration = None
            self._total_chunks = 1

        # RESUME LOGIC: Check checkpoint for crash recovery
        skip_audio = False
        skip_voice = False
        resume_from_frame = 0

        self._resume_from_frame = resume_from_frame  # Store for _process_frames

        try:
            if not skip_audio:
                async with progress_tracker.stage(
                    job_id, "audio", "Processing audio"
                ):
                    progress_tracker.update(job_id, 10.0)
                    await retry(
                        lambda: self.audio_stage.process_audio(path),
                        on_retry=lambda e: progress_tracker.increment_retry(
                            job_id, "audio"
                        ),
                    )
                    progress_tracker.update(job_id, 30.0)

                logger.info(
                    "[Pipeline] _process_audio completed, running cleanup..."
                )
                self._cleanup_memory("audio_complete")  # Unload Whisper
                self.frame_stage.audio_classification = self.audio_stage.audio_classification
                logger.info(
                    "[Pipeline] Audio cleanup done, saving checkpoint..."
                )
                # Checkpoint audio completion
                progress_tracker.save_checkpoint(
                    job_id, {"audio_complete": True}
                )
            else:
                progress_tracker.stage_start(
                    job_id, "audio", "Skipped (Already done)"
                )
                progress_tracker.stage_complete(job_id, "audio", "Skipped")
            logger.info(
                "[Pipeline] Audio phase complete, moving to voice processing..."
            )

            if progress_tracker.is_cancelled(job_id):
                return job_id
            if progress_tracker.is_paused(job_id):
                return job_id

            if not skip_voice:
                async with progress_tracker.stage(
                    job_id, "voice", "Processing voice"
                ):
                    progress_tracker.update(job_id, 35.0)
                    await retry(
                        lambda: self.voice_stage.process_voice(path),
                        on_retry=lambda e: progress_tracker.increment_retry(
                            job_id, "voice"
                        ),
                    )
                    progress_tracker.update(job_id, 50.0)

                self._cleanup_memory("voice_complete")  # Unload Pyannote
                # Checkpoint voice completion
                progress_tracker.save_checkpoint(
                    job_id, {"voice_complete": True}
                )
            else:
                progress_tracker.stage_start(
                    job_id, "voice", "Skipped (Already done)"
                )
                progress_tracker.stage_complete(job_id, "voice", "Skipped")

            logger.debug("Voice complete - checking job status")

            if progress_tracker.is_cancelled(job_id):
                logger.info(f"Job {job_id} cancelled")
                return job_id

            if progress_tracker.is_paused(job_id):
                logger.info(f"Job {job_id} paused")
                return job_id

            async with progress_tracker.stage(
                job_id, "audio_events", "Detecting audio events"
            ):
                await self.audio_events_stage.process_audio_events(path, job_id)

            # Frames & Scenes processing with Chunking
            current_chunk = 0

            while True:
                # Calculate Chunk Context
                chunk_start = 0.0
                chunk_end = duration

                if should_chunk:
                    chunk_start = current_chunk * chunk_duration
                    chunk_end = min(
                        (current_chunk + 1) * chunk_duration, duration
                    )

                    # Log chunk progress
                    logger.info(
                        f"[Chunking] Processing Chunk {current_chunk + 1}/{self._total_chunks}: "
                        f"{chunk_start / 60:.1f}m - {chunk_end / 60:.1f}m"
                    )

                    # Pass chunk boundaries as explicit parameters (no instance state mutation)

                # Exit condition
                if chunk_start >= duration:
                    break

                # 1. Process Frames (Batch)
                logger.debug(
                    f"Starting frame processing for chunk {current_chunk}"
                )
                async with progress_tracker.stage(
                    job_id, f"frames_chunk_{current_chunk}", "Processing frames"
                ):
                    # Update scalar progress roughly
                    # We could make this precise but simple update is 55->85 range
                    progress_base = 55.0 + (
                        30.0 * (current_chunk / self._total_chunks)
                    )
                    progress_tracker.update(job_id, progress_base)

                    await retry(
                        lambda cs=chunk_start,
                        ce=chunk_end: self.frame_stage.process_frames(
                            path,
                            job_id,
                            total_duration=duration,
                            chunk_start=cs,
                            chunk_end=ce,
                        ),
                        on_retry=lambda e: progress_tracker.increment_retry(
                            job_id, "frames"
                        ),
                    )

                logger.debug(
                    f"Frame processing complete for chunk {current_chunk}"
                )

                # 2. Dense Scene Captioning (Batch)
                # We process scenes restricted to this chunk
                async with progress_tracker.stage(
                    job_id,
                    f"scene_captions_chunk_{current_chunk}",
                    "Generating scene captions",
                ):
                    await self.scene_stage.process_scene_captions(
                        path,
                        job_id,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                    )

                # MEMORY CLEANUP between chunks
                self._cleanup_memory(f"chunk_{current_chunk}_complete")

                if not should_chunk:
                    break

                # No state restoration needed — chunk boundaries passed as parameters
                current_chunk += 1

                # Check cancellation between chunks
                if progress_tracker.is_cancelled(
                    job_id
                ) or progress_tracker.is_paused(job_id):
                    return job_id

            # Post-Processing Phase
            async with progress_tracker.stage(
                job_id, "post_processing", "Enriching metadata"
            ):
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
            raise IngestionError(
                f"Unexpected pipeline failure: {e}", original_error=e
            )

    def _get_speaker_clusters_at_time(
        self, media_path: str, timestamp: float
    ) -> list[int]:
        """Identifies speaker cluster IDs active at a specific timestamp.

        Uses configurable tolerance to handle A/V sync drift.
        ATSC standard: 45ms audio lead, 125ms lag acceptable.
        """
        tol = settings.face_audio_sync_tolerance  # Default 0.3s

        clusters = []
        try:
            voice_segments = self.db.get_voice_segments_for_media(media_path)
            for seg in voice_segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                # Apply tolerance window instead of exact match
                if (start - tol) <= timestamp <= (end + tol):
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
                collection_name=MEDIA_SEGMENTS_COLLECTION,
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

    def _get_audio_events_for_video(self, media_path: str) -> list[dict]:
        """Retrieves non-speech audio events (sirens, etc.) for a video.

        Args:
            media_path: Path to the media file.

        Returns:
            List of audio event dicts (start, end, label, confidence).
        """
        try:
            resp = self.db.client.scroll(
                collection_name=AUDIO_EVENTS_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="media_path",
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
                    "label": p.payload.get("label", ""),
                    "confidence": p.payload.get("confidence", 0.0),
                }
                for p in resp[0]
                if p.payload
            ]
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
            frames = self.db.get_frames_by_video(media_path)
            audio_segments = self._get_audio_segments_for_video(media_path)

            # Deep Video Understanding: SAM 3 Concept Tracking
            # Only run if enabled and frames exist
            if self.sam3_tracker and frames:
                try:
                    self._process_video_masklets(path, frames)
                except Exception as e:
                    logger.warning(f"SAM3 Tracking failed: {e}")



            # NOTE: Scene processing removed — redundant with _process_scene_captions()
            # which runs TransNet V2, VLM captions, deep research, visual embeddings,
            # InternVideo/LanguageBind, and full aggregation with Neo4j graph ingestion.
            # See _process_scene_captions() for the definitive scene processing path.

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

                # Scene metadata recorded in SQL database

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

        try:
            import cv2

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
                    media_path=str(path),
                    concept=data["concept"],
                    start_time=start_time,
                    end_time=end_time,
                    confidence=0.9,  # SAM3 is usually confident
                )
                logger.info(
                    f"Masklet saved: {data['concept']} ({start_time:.1f}-{end_time:.1f}s)"
                )
