"""Media ingestion pipeline orchestrator."""

from __future__ import annotations

import asyncio
import gc
import uuid
from pathlib import Path
from typing import Any, Iterable

import torch
from qdrant_client.http import models

from config import settings
from core.llm.vlm_factory import get_vlm_client
from core.processing.extractor import FrameExtractor
from core.processing.identity import FaceManager, FaceTrackBuilder
from core.processing.metadata import MetadataEngine
from core.processing.prober import MediaProbeError, MediaProber
from core.processing.scene_detector import detect_scenes, extract_scene_frame
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
        """Initialize the pipeline and its sub-components."""
        self.prober = MediaProber()
        self.extractor = FrameExtractor()
        self.db = VectorDB(
            backend=qdrant_backend,
            host=qdrant_host,
            port=qdrant_port,
        )
        self.metadata_engine = MetadataEngine(tmdb_api_key=tmdb_api_key)
        self.frame_interval_seconds = frame_interval_seconds
        self.vision: VisionAnalyzer | None = None
        self.faces: FaceManager | None = None
        self.voice: VoiceProcessor | None = None
        self.frame_sampler = FrameSampler(every_n=5)

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
    ):
        """Initialize the media ingestion pipeline."""
        resume = False
        if job_id:
            resume = True
        else:
            job_id = str(uuid.uuid4())

        bind_context(component="pipeline")

        # Store time range for use in processing methods
        self._start_time = start_time
        self._end_time = end_time

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

        _ = self.metadata_engine.identify(path, user_hint=hint_enum)

        try:
            probed = self.prober.probe(path)
            duration = float(probed.get("format", {}).get("duration", 0.0))
        except MediaProbeError as e:
            progress_tracker.fail(job_id, error=f"Media probe failed: {e}")
            raise

        try:
            progress_tracker.update(job_id, 5.0, stage="audio", message="Processing audio")
            await retry(lambda: self._process_audio(path))
            self._cleanup_memory("audio_complete")  # Unload Whisper
            progress_tracker.update(job_id, 30.0, stage="audio", message="Audio complete")

            if progress_tracker.is_cancelled(job_id):
                return job_id
            if progress_tracker.is_paused(job_id):
                return job_id

            progress_tracker.update(job_id, 35.0, stage="voice", message="Processing voice")
            await retry(lambda: self._process_voice(path))
            self._cleanup_memory("voice_complete")  # Unload Pyannote
            progress_tracker.update(job_id, 50.0, stage="voice", message="Voice complete")

            with open("debug_flow.log", "a") as f: f.write("Voice complete. Checking cancelled...\n")

            if progress_tracker.is_cancelled(job_id):
                with open("debug_flow.log", "a") as f: f.write("Cancelled.\n")
                return job_id

            with open("debug_flow.log", "a") as f: f.write("Checking paused...\n")
            if progress_tracker.is_paused(job_id):
                with open("debug_flow.log", "a") as f: f.write("Paused.\n")
                return job_id

            with open("debug_flow.log", "a") as f: f.write("Starting frames...\n")
            progress_tracker.update(job_id, 55.0, stage="frames", message="Processing frames")

            with open("debug_flow.log", "a") as f: f.write("Calling _process_frames...\n")
            await retry(lambda: self._process_frames(path, job_id, total_duration=duration))

            with open("debug_flow.log", "a") as f: f.write("Frames complete cleanup...\n")
            self._cleanup_memory("frames_complete")

            if progress_tracker.is_cancelled(job_id) or progress_tracker.is_paused(job_id):
                return job_id

            # Dense Scene Captioning (VLM on detected scene boundaries)
            progress_tracker.update(job_id, 90.0, stage="scene_captions", message="Generating scene captions")
            await self._process_scene_captions(path, job_id)

            progress_tracker.complete(job_id, message=f"Completed: {path.name}")

            return job_id

        except Exception as e:
            progress_tracker.fail(job_id, error=str(e))
            raise

    @observe("audio_processing")
    async def _process_audio(self, path: Path) -> None:
        """Process audio with auto language detection and AI4Bharat for Indic languages."""
        from core.utils.logger import log

        audio_segments: list[dict[str, Any]] = []
        srt_path = path.with_suffix(".srt")

        # Check for existing sidecar SRT
        if srt_path.exists():
            audio_segments = parse_srt(srt_path) or []
            if audio_segments:
                log(f"[Audio] Using existing SRT: {len(audio_segments)} segments")

        # Check for embedded subtitles
        if not audio_segments:
            await resource_manager.throttle_if_needed("compute")
            try:
                with AudioTranscriber() as transcriber:
                    temp_srt = path.with_suffix(".embedded.srt")
                    if transcriber._find_existing_subtitles(path, temp_srt, None, "ta"):
                        audio_segments = parse_srt(temp_srt) or []
                        if audio_segments:
                            log(f"[Audio] Extracted embedded subs: {len(audio_segments)} segments")
                        if temp_srt.exists():
                            temp_srt.unlink()
            except Exception as e:
                log(f"[Audio] Embedded subtitle extraction failed: {e}")

        # Run ASR if no existing subtitles
        if not audio_segments:
            await resource_manager.throttle_if_needed("compute")

            # Auto-detect language if enabled
            detected_lang = "en"
            if settings.auto_detect_language:
                detected_lang = await self._detect_audio_language(path)
                log(f"[Audio] Detected language: {detected_lang}")
            else:
                detected_lang = settings.language or "en"

            # Choose transcriber based on language
            indic_languages = ["ta", "hi", "te", "ml", "kn", "bn", "gu", "mr", "or", "pa"]

            if settings.use_indic_asr and detected_lang in indic_languages:
                # Use AI4Bharat for Indic languages
                log(f"[Audio] Attempting AI4Bharat IndicConformer for '{detected_lang}'")
                indic_transcriber = None
                try:
                    from core.processing.indic_transcriber import IndicASRPipeline
                    indic_transcriber = IndicASRPipeline(lang=detected_lang)
                    log(f"[Audio] IndicASR backend: {indic_transcriber._backend}")
                    
                    # Generate SRT sidecar file alongside the video
                    srt_path = path.with_suffix(".srt")
                    audio_segments = indic_transcriber.transcribe(path, output_srt=srt_path) or []
                    
                    if audio_segments:
                        log(f"[Audio] AI4Bharat SUCCESS: {len(audio_segments)} segments")
                        log(f"[Audio] SRT saved to: {srt_path}")
                    else:
                        log("[Audio] AI4Bharat returned empty, falling back to Whisper")
                        raise ValueError("AI4Bharat returned no segments")
                except Exception as e:
                    log(f"[Audio] AI4Bharat failed: {e}")
                    log(f"[Audio] Falling back to Whisper for '{detected_lang}'")
                    try:
                        with AudioTranscriber() as transcriber:
                            audio_segments = transcriber.transcribe(path, language=detected_lang) or []
                    except Exception as e2:
                        log(f"[Audio] Whisper fallback also failed: {e2}")
                finally:
                    # CRITICAL: Unload IndicASR to free VRAM for Ollama vision
                    if indic_transcriber is not None:
                        indic_transcriber.unload_model()
            else:
                # Use Whisper for English and other languages
                log(f"[Audio] Using Whisper turbo for '{detected_lang}'")
                try:
                    with AudioTranscriber() as transcriber:
                        audio_segments = transcriber.transcribe(path, language=detected_lang) or []
                except Exception as e:
                    log(f"[Audio] Whisper failed: {e}")

            if audio_segments:
                log(f"[Audio] Transcription SUCCESS: {len(audio_segments)} segments")
            else:
                log(f"[Audio] WARNING - NO SEGMENTS produced for {path.name}")
                # NEVER-EMPTY GUARANTEE: Create a placeholder segment to preserve timeline
                # This ensures search can still find the media by path/timestamp
                try:
                    probed = self.prober.probe(path)
                    duration = float(probed.get("format", {}).get("duration", 0.0))
                    if duration > 0:
                        audio_segments = [{
                            "text": "[No speech detected]",
                            "start": 0.0,
                            "end": duration,
                        }]
                        log(f"[Audio] Created placeholder segment for {duration:.1f}s media")
                except Exception:
                    pass

        if audio_segments:
            prepared = self._prepare_segments_for_db(path=path, chunks=audio_segments)
            self.db.insert_media_segments(str(path), prepared)
            log(f"[Audio] Stored {len(prepared)} dialogue segments in DB")

        self._cleanup_memory()

    async def _detect_audio_language(self, path: Path) -> str:
        """Detect audio language using robust AudioTranscriber.
        
        Returns:
            ISO 639-1 language code (e.g., 'en', 'ta', 'hi')
        """
        from core.processing.transcriber import AudioTranscriber
        
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
        """Helper to run detection synchronously."""
        with AudioTranscriber() as transcriber:
            return transcriber.detect_language(path)

    @observe("voice_processing")
    async def _process_voice(self, path: Path) -> None:
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

            for idx, seg in enumerate(voice_segments or []):
                audio_path: str | None = None
                if seg.embedding is not None:
                    # Extract audio clip
                    try:
                        clip_name = f"{safe_stem}_{seg.start_time:.2f}_{seg.end_time:.2f}.mp3"
                        clip_file = thumb_dir / clip_name

                        if not clip_file.exists():
                            cmd = [
                                "ffmpeg",
                                "-y",
                                "-i", str(path),
                                "-ss", str(seg.start_time),
                                "-to", str(seg.end_time),
                                "-q:a", "2",  # High quality MP3
                                "-map", "a",
                                str(clip_file)
                            ]
                            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                        if clip_file.exists():
                            audio_path = f"/thumbnails/voices/{clip_name}"
                    except Exception:
                        pass

                    self.db.insert_voice_segment(
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        speaker_label=seg.speaker_label,
                        embedding=seg.embedding,
                        audio_path=audio_path,
                    )
        finally:
            del self.voice
            self.voice = None
            self._cleanup_memory()

    @observe("frame_processing")
    async def _process_frames(self, path: Path, job_id: str | None = None, total_duration: float = 0.0) -> None:
        vision_task_type = "network" if settings.llm_provider == "gemini" else "compute"
        await resource_manager.throttle_if_needed(vision_task_type)

        # Use the configured LLM provider from settings
        from llm.factory import LLMFactory
        vision_llm = LLMFactory.create_llm(provider=settings.llm_provider.value)
        self.vision = VisionAnalyzer(llm=vision_llm)
        # InsightFace uses GPU for fast detection, but we unload it before Ollama
        self.faces = FaceManager(use_gpu=settings.device == "cuda")

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
            start_time=getattr(self, '_start_time', None),
            end_time=getattr(self, '_end_time', None),
        )

        # Time offset for accurate timestamps in partial extraction
        time_offset = getattr(self, '_start_time', None) or 0.0

        frame_count = 0

        # Sliding window context for narrative continuity (FAANG-level)
        # 5-frame window captures motion, actions, and temporal relationships
        # This is critical for video-aware search (not treating frames as isolated images)
        from collections import deque
        context_window: deque[tuple[float, str]] = deque(maxlen=5)  # (timestamp, description)

        async for frame_path in frame_generator:
            if job_id:
                if progress_tracker.is_cancelled(job_id) or progress_tracker.is_paused(job_id):
                    break

            # Calculate actual timestamp including offset
            timestamp = time_offset + (frame_count * float(self.frame_interval_seconds))
            try:
                if self.frame_sampler.should_sample(frame_count):
                    # Build narrative context from sliding window for temporal awareness
                    narrative_parts = [f"[{ts:.1f}s] {desc[:150]}" for ts, desc in context_window]
                    narrative_context = " -> ".join(narrative_parts) if narrative_parts else "Start of video"

                    # Pass neighbor timestamps for temporal embedding
                    neighbor_timestamps = [ts for ts, _ in context_window]

                    new_desc = await self._process_single_frame(
                        video_path=path,
                        frame_path=frame_path,
                        timestamp=timestamp,
                        index=frame_count,
                        context=narrative_context,
                        neighbor_timestamps=neighbor_timestamps,
                    )
                    if new_desc:
                        # Store timestamp with description for temporal relationships
                        context_window.append((timestamp, new_desc[:200]))

                if job_id:
                    # Check for PAUSE functionality
                    if progress_tracker.is_paused(job_id):
                        logger.info(f"Job {job_id} paused. Stopping frame loop.")
                        break # Exit loop (checkpoint saved in update_granular)

                    if progress_tracker.is_cancelled(job_id):
                        break

                    # Update Granular Stats
                    # Use provided duration for 100% accuracy, fallback to metadata
                    video_duration = total_duration
                    if not video_duration:
                        try:
                            probe_data = self.prober.probe(path)
                            video_duration = float(probe_data.get("format", {}).get("duration", 0.0))
                        except Exception:
                            video_duration = 0.0

                    interval = float(self.frame_interval_seconds)

                    total_est_frames = int(video_duration / interval) if video_duration else 0
                    current_ts = timestamp
                    current_frame_index = int(current_ts / interval) if interval > 0 else frame_count

                    status_msg = f"Processing frame {current_frame_index}/{total_est_frames} at {current_ts:.1f}s"

                    progress_tracker.update_granular(
                        job_id,
                        processed_frames=current_frame_index,
                        total_frames=total_est_frames,
                        current_timestamp=current_ts,
                        total_duration=video_duration
                    )

                    if frame_count % 5 == 0:
                        progress = 55.0 + min(40.0, (current_ts / (video_duration or 1)) * 40.0) if video_duration else 55.0
                        progress_tracker.update(
                            job_id,
                            progress,
                            stage="frames",
                            message=status_msg,
                        )

                await asyncio.sleep(0.01)  # Minimal sleep for responsiveness
            finally:
                # Always delete the frame file after processing
                if frame_path.exists():
                    try:
                        frame_path.unlink()
                    except Exception:
                        pass

            frame_count += 1

            # Aggressive memory cleanup every 5 frames to prevent OOM (same approach as audio chunking)
            # This preserves timestamps and accuracy while managing VRAM on 8GB GPUs
            CLEANUP_INTERVAL = 5
            if frame_count % CLEANUP_INTERVAL == 0:
                self._cleanup_memory(context=f"frame_{frame_count}")
                # Extra VRAM flush for video processing (mirrors audio chunking)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Finalize face tracks and store in Identity Graph
        # This is the key step: convert frame-by-frame detections into stable tracks
        if hasattr(self, '_face_track_builder') and self._face_track_builder:
            try:
                finalized_tracks = self._face_track_builder.finalize_all()
                logger.info(f"Finalized {len(finalized_tracks)} face tracks for {path.name}")

                # Store each track in the Identity Graph
                media_id = getattr(self, '_current_media_id', str(path))
                for track_id, avg_embedding, metadata in self._face_track_builder.get_track_embeddings():
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
                        logger.warning(f"Failed to store face track: {track_err}")
            except Exception as e:
                logger.warning(f"Track finalization failed: {e}")

        # Final cleanup
        del self.vision
        del self.faces
        self.vision = None
        self.faces = None
        if hasattr(self, '_face_track_builder'):
            del self._face_track_builder
        self._cleanup_memory()

    @observe("scene_captioning")
    async def _process_scene_captions(self, path: Path, job_id: str | None = None) -> None:
        """Process scenes and store with multi-vector embeddings.
        
        This is the production-grade approach:
        1. Detect scene boundaries
        2. Aggregate frame data per scene (entities, actions, faces, clothing)
        3. Get dialogue for each scene from audio segments
        4. Store scene with visual/motion/dialogue vectors
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
                    caption = vlm.generate_caption_from_bytes(frame_bytes, prompt)
                    if caption:
                        visual_summary = caption
                except Exception as e:
                    logger.warning(f"VLM caption failed for scene {idx}: {e}")

            # 2. Filter frames that belong to this scene
            scene_frames = [
                f for f in all_frames
                if scene.start_time <= f.get("timestamp", 0) <= scene.end_time
            ]

            # 3. Filter audio segments that overlap this scene
            scene_audio = [
                a for a in audio_segments
                if a.get("end", 0) > scene.start_time and a.get("start", 0) < scene.end_time
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
                    visual_parts.append(f"{attr['clothing_color']} {attr['clothing_type']}")
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
                        e.get("name", "") for e in aggregated.get("entities", [])
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

        logger.info(f"Stored {scenes_stored}/{len(scenes)} scenes for {path.name}")

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
        """Process a single frame with face-frame linking and structured analysis.
        
        Order: Faces FIRST -> Vision Analysis -> Store with linked identities
        
        Uses incremental face clustering (not hash-based) for consistent identity.
        Stores temporal context (neighbor_timestamps) for video-aware search.
        """
        if not self.vision or not self.faces:
            return None

        # Track cluster centroids across frames (initialized once per video)
        if not hasattr(self, '_face_clusters'):
            self._face_clusters: dict[int, list[float]] = {}

        # 1. DETECT FACES FIRST (Capture Identity before Vision)
        face_cluster_ids: list[int] = []
        detected_faces = []
        try:
            detected_faces = await self.faces.detect_faces(frame_path)
        except Exception:
            pass

        # Feed detected faces into FaceTrackBuilder for temporal grouping
        # This creates stable per-video tracks before global identity linking
        if hasattr(self, '_face_track_builder') and detected_faces:
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
                # PROPER CLUSTERING: Use embedding similarity, not hash
                cluster_id, self._face_clusters = self.faces.match_or_create_cluster(
                    embedding=face.embedding,
                    existing_clusters=self._face_clusters,
                    threshold=settings.face_clustering_threshold,
                )
                face_cluster_ids.append(cluster_id)

                # Crop and save face thumbnail with better quality
                thumb_path: str | None = None
                try:
                    import cv2
                    import numpy as np
                    img_data = np.fromfile(str(frame_path), dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is not None:
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
                                scale = max(min_size / crop_h, min_size / crop_w)
                                new_w = int(crop_w * scale)
                                new_h = int(crop_h * scale)
                                face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                            # Use slightly lower quality (95) to save memory/space, explicit int cast for resize
                            thumb_name = f"{safe_stem}_{timestamp:.2f}_{idx}.jpg"
                            thumb_file = thumb_dir / thumb_name

                            try:
                                # Pre-check memory availability or just robust try/catch
                                cv2.imwrite(str(thumb_file), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                thumb_path = f"/thumbnails/faces/{thumb_name}"
                            except cv2.error as e:
                                logger.warning(f"Thumbnail save skipped (OpenCV error): {e}")
                                # Clean up if partial write happened
                                if thumb_file.exists() and thumb_file.stat().st_size == 0:
                                    thumb_file.unlink()

                except (MemoryError, cv2.error) as e:
                    logger.warning(f"Thumbnail generation skipped (OOM/CV error): {e}")
                    gc.collect() # Try to recover leaks
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
                    bbox_size=getattr(face, '_bbox_size', None),
                    det_score=face.confidence if hasattr(face, 'confidence') else None,
                )

        # 2. RUN VISION ANALYSIS (With OCR and structured output)
        description: str | None = None
        analysis = None

        # CRITICAL: Unload face models to free VRAM for Ollama
        # This releases the ~1GB held by InsightFace/ArcFace so llava:7b can fit
        if self.faces:
            self.faces.unload_gpu()

        try:
            # Use structured analysis that outputs FrameAnalysis
            analysis = await self.vision.analyze_frame(frame_path)
            if analysis:
                # Generate searchable text with specific terms (Idly not food)
                description = analysis.to_search_content()
                # Inject face cluster IDs into analysis
                analysis.face_ids = [str(cid) for cid in face_cluster_ids]
        except Exception as e:
            logger.warning(f"Structured analysis failed: {e}, falling back to describe")

        # Fallback to unstructured description
        if not description:
            try:
                description = await self.vision.describe(frame_path, context=context)
            except Exception:
                pass

        # 3. STORE FRAME WITH LINKED IDENTITIES AND STRUCTURED PAYLOAD
        if description:
            vector = self.db.encoder.encode(description).tolist()

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
                current_face_names = {} # cluster_id -> name
                for cid in face_cluster_ids:
                    fname = self.db.get_face_name_by_cluster(cid)
                    if fname:
                        current_face_names[cid] = fname
                        payload["face_names"].append(fname)

                # Now process speaker clusters
                for cluster_id in speaker_cluster_ids:
                    speaker_name = self.db.get_speaker_name_by_cluster(cluster_id)

                    if speaker_name:
                        payload["speaker_names"].append(speaker_name)
                        # Propagate Speaker Name -> Unnamed Faces
                        if not current_face_names and face_cluster_ids:
                             # Heuristic: If there's exactly one unnamed face and one named speaker, link them
                             if len(face_cluster_ids) == 1:
                                 face_cid = face_cluster_ids[0]
                                 logger.info(f"Auto-mapping Speaker '{speaker_name}' -> Face Cluster {face_cid}")
                                 self.db.set_face_name(face_cid, speaker_name)
                                 payload["face_names"].append(speaker_name) # Update current payload

                    elif current_face_names:
                        # Propagate Face Name -> Unnamed Speaker
                        # Heuristic: If exactly one named face is visible, assume they are the speaker
                        if len(current_face_names) == 1:
                            face_name = list(current_face_names.values())[0]
                            logger.info(f"Auto-mapping Face '{face_name}' -> Speaker Cluster {cluster_id}")
                            self.db.set_speaker_name(cluster_id, face_name)
                            payload["speaker_names"].append(face_name) # Update current payload

            except Exception as e:
                logger.warning(f"Face-Audio mapping error: {e}")

            # 3b. (Skipped redundant loop, handled above)

            # 3c. Build identity text for searchability
            identity_parts = []
            if payload["face_names"]:
                identity_parts.append(f"Visible: {', '.join(payload['face_names'])}")
            if payload["speaker_names"]:
                identity_parts.append(f"Speaking: {', '.join(payload['speaker_names'])}")
            if identity_parts:
                payload["identity_text"] = ". ".join(identity_parts)

            # Add structured data if available for hybrid search
            if analysis:
                payload["structured_data"] = analysis.model_dump()
                payload["visible_text"] = analysis.scene.visible_text if analysis.scene else []
                payload["entities"] = [e.name for e in analysis.entities] if analysis.entities else []
                payload["entity_categories"] = list({e.category for e in analysis.entities}) if analysis.entities else []
                payload["scene_location"] = analysis.scene.location if analysis.scene else ""
                payload["action"] = analysis.action or ""
                payload["description"] = description

            # 3d. Add temporal context for video-aware search (not isolated frames)
            # This enables queries like "pin falling slowly" by connecting adjacent frames
            if neighbor_timestamps:
                payload["neighbor_timestamps"] = neighbor_timestamps
                payload["temporal_window_size"] = len(neighbor_timestamps)
            if context:
                # Store truncated context for retrieval boost
                payload["temporal_context"] = context[:500] if len(context) > 500 else context

            # 3e. Generate Vector (Include identity for searchability)
            # "A man walking. Visible: John. Speaking: John"
            full_text = description
            if "identity_text" in payload:
                full_text = f"{description}. {payload['identity_text']}"

            vector = self.db.encoder.encode(full_text).tolist()

            self.db.upsert_media_frame(
                point_id=f"{video_path}_{timestamp:.3f}",
                vector=vector,
                video_path=str(video_path),
                timestamp=timestamp,
                action=description,
                payload=payload,
            )

        return description

    def _get_speaker_clusters_at_time(self, media_path: str, timestamp: float) -> list[int]:
        """Get speaker cluster IDs who are speaking at a given timestamp.
        
        This enables face-audio mapping by finding which speaker is talking
        when a particular face is visible.
        
        Args:
            media_path: Path to the media file.
            timestamp: Frame timestamp in seconds.
            
        Returns:
            List of speaker cluster IDs who are speaking at this time.
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
        """Get all audio/dialogue segments for a video.
        
        Used by scene aggregation to include dialogue in scene data.
        
        Args:
            media_path: Path to the media file.
            
        Returns:
            List of audio segment dicts with start, end, text.
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
                for p in resp[0] if p.payload
            ]
        except Exception:
            return []

    def _get_frames_for_video(self, media_path: str) -> list[dict]:
        """Get all indexed frames for a video.
        
        Used by scene aggregation to collect frame data per scene.
        
        Args:
            media_path: Path to the media file.
            
        Returns:
            List of frame dicts with timestamp, payload data.
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
                    frames.append({
                        "id": str(p.id),
                        "timestamp": p.payload.get("timestamp", 0),
                        "action": p.payload.get("action", ""),
                        "description": p.payload.get("description", "") or p.payload.get("action", ""),
                        "face_cluster_ids": p.payload.get("face_cluster_ids", []),
                        "face_names": p.payload.get("face_names", []),
                        "speaker_names": p.payload.get("speaker_names", []),
                        "visible_text": p.payload.get("visible_text", []),
                        "entities": p.payload.get("entities", []),
                        "structured_data": p.payload.get("structured_data", {}),
                        "scene_location": p.payload.get("scene_location", ""),
                        "scene_cultural": p.payload.get("scene_cultural", ""),
                    })
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
