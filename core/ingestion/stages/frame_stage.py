"""Frame extraction, OCR, object detection, and visual analysis stage.

Extracted from IngestionPipeline to reduce God class size.
IngestionPipeline inherits from FrameStageMixin to compose these methods.
"""

from __future__ import annotations

import asyncio
import gc
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from config import settings
from core.storage.db import VectorDB
from core.utils.logger import logger
from core.utils.progress import progress_tracker

if TYPE_CHECKING:
    pass


class FrameStageMixin:
    """Frame extraction, OCR, object detection, and visual analysis stage."""

    # These will be available via IngestionPipeline inheritance
    db: VectorDB
    # Type stub for type checkers (allows accessing self.* in mixin)
    def __getattr__(self, name: str) -> Any: ...

    async def _process_frames(
        self, path: Path, job_id: str | None = None, total_duration: float = 0.0,
        chunk_start: float | None = None, chunk_end: float | None = None,
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
        from core.processing.extractor import FrameExtractor
        from llm.factory import LLMFactory

        vision_llm = LLMFactory.create_llm(provider=settings.llm_provider.value)
        self.vision = VisionAnalyzer(llm=vision_llm)

        # Initialize Visual Encoder for Search Embeddings (SigLIP/CLIP)
        from core.processing.visual_encoder import get_default_visual_encoder

        self.visual_encoder = get_default_visual_encoder()

        # GLOBAL IDENTITY: Load existing cluster centroids from DB
        # This enables cross-video identity matching (O(1) gallery-probe)
        global_clusters = self.db.get_all_cluster_centroids()
        logger.info(
            f"[GlobalIdentity] Loaded {len(global_clusters)} cluster centroids for matching"
        )

        # InsightFace uses GPU for fast detection, but we unload it before Ollama
        self.faces = FaceManager(
            dbscan_eps=settings.hdbscan_cluster_selection_epsilon,
            dbscan_min_samples=settings.hdbscan_min_samples,
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
                start_time=chunk_start,
                end_time=chunk_end,
                output_dir=frame_cache_dir,
            )

            # Resume support: skip already processed frames
            resume_from_frame = getattr(self, "_resume_from_frame", 0)

            # XMem-style temporal context for video coherence
            from core.processing.temporal_context import (
                TemporalContext,
            )

            temporal_ctx = TemporalContextManager(sensory_size=5)

            # Scenelet Builder (Sliding Window: Optimized via Config)
            scenelet_builder = SceneletBuilder(
                window_seconds=settings.scenelet_window_seconds,
                stride_seconds=settings.scenelet_stride_seconds,
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
                    if getattr(settings, "enable_face_recognition", True):
                        batch_faces = await self.faces.detect_faces_batch(paths)
                    else:
                        batch_faces = [[] for _ in paths]
                except Exception as e:
                    logger.warning(f"Batch face detection failed: {e}")
                    batch_faces = [None] * len(frames_to_process)

                # === 2. BATCH VISUAL ENCODING (NEW - Major Speedup) ===
                batch_embeddings = [None] * len(frames_to_process)
                try:
                    if self.visual_encoder:
                        # Use batch encoding (SigLIP/CLIP) for search inputs
                        embeddings = await self.visual_encoder.encode_batch(
                            paths
                        )
                        batch_embeddings = embeddings
                        logger.debug(
                            f"[Vision] Batch encoded {len(paths)} frames"
                        )
                    elif self.vision and hasattr(self.vision, "encode_batch"):
                        # Fallback to VLM if no dedicated encoder (rare)
                        embeddings = await self.vision.encode_batch(paths)
                        batch_embeddings = embeddings
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
                    neighbor_timestamps = [
                        c.timestamp for c in temporal_ctx.sensory
                    ]

                    new_desc = await self._process_single_frame(
                        video_path=path,
                        frame_path=f_path,
                        timestamp=f_ts,
                        index=f_idx,
                        context=narrative_context,
                        neighbor_timestamps=neighbor_timestamps,
                        pre_detected_faces=batch_faces[idx]
                        if idx < len(batch_faces)
                        else None,
                        pre_computed_embedding=batch_embeddings[idx]
                        if idx < len(batch_embeddings)
                        else None,
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
                    # Use config batch_size for optimal hardware utilization
                    if len(pending_frames) >= settings.batch_size:
                        await _process_batch_items(pending_frames)
                        pending_frames = []

                if job_id:
                    if progress_tracker.is_paused(job_id):
                        logger.info(
                            f"Job {job_id} paused. Stopping frame loop."
                        )
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
                            probe_data = await self.get_probe_data(path)
                            video_duration = float(
                                probe_data.get("format", {}).get(
                                    "duration", 0.0
                                )
                            )
                        except Exception:
                            video_duration = 0.0

                    interval = float(self.frame_interval_seconds)

                    total_est_frames = (
                        int(video_duration / interval) if video_duration else 0
                    )
                    current_ts = timestamp
                    current_frame_index = (
                        int(current_ts / interval)
                        if interval > 0
                        else frame_count
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

                await asyncio.sleep(0)  # Yield event loop without wall-clock delay
                # Always delete the frame file after processing
                # Only delete if NOT in pending batch (processed frames are deleted by helper)
                if extracted_frame not in pending_frames:
                    if frame_path.exists():
                        try:
                            frame_path.unlink()
                        except Exception:
                            pass

                # frame_count is already set from extracted_frame.frame_index

                # Periodic memory cleanup to prevent OOM without stalling GPU
                # synchronize() blocks the pipeline — use empty_cache() only
                cleanup_interval = 20
                if frame_count % cleanup_interval == 0:
                    self._cleanup_memory(context=f"frame_{frame_count}")

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
            if (
                hasattr(self, "_face_track_builder")
                and self._face_track_builder
            ):
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
                                avg_confidence=metadata.get(
                                    "avg_confidence", 0.0
                                ),
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
        if getattr(settings, "enable_deep_research", True):
            skip_deep_research = getattr(
                settings, "deep_research_per_scene", True
            )
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

        safe_stem = hashlib.md5(video_path.stem.encode()).hexdigest()

        for idx, face in enumerate(detected_faces):
            if face.embedding is not None:
                # GLOBAL IDENTITY: Match against DB clusters (gallery-probe)
                # FaceManager.global_clusters loaded at _process_frames init
                # Lock prevents race condition when multiple frames are processed concurrently
                async with self._face_cluster_lock:
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

                frame_data = np.fromfile(str(frame_path), dtype=np.uint8)
                frame_img = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

                if frame_img is not None:
                    # --- OCR Skip-Unchanged-Frames Optimization ---
                    skip_ocr = False

                    # 1. Time Throttling (Max every 2.0s)

                    now_ts = time.time()
                    if (
                        hasattr(self, "_last_ocr_time")
                        and (now_ts - self._last_ocr_time) < 2.0
                    ):
                        skip_ocr = True

                    # 2. Perceptual Hash Check
                    ocr_skip_enabled = getattr(
                        settings, "ocr_skip_unchanged_frames", True
                    )

                    if not skip_ocr and ocr_skip_enabled:
                        try:
                            # Compute perceptual hash (fast, 8x8 grayscale downsample)
                            gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
                            resized = cv2.resize(
                                gray, (8, 8), interpolation=cv2.INTER_AREA
                            )
                            mean_val = np.mean(resized)
                            current_hash = (
                                (resized > mean_val).flatten().tobytes()
                            )

                            # Compare with previous frame's hash
                            if (
                                hasattr(self, "_last_ocr_hash")
                                and self._last_ocr_hash is not None
                            ):
                                # Hamming distance (count of differing bits)
                                diff = sum(
                                    a != b
                                    for a, b in zip(
                                        current_hash,
                                        self._last_ocr_hash,
                                        strict=True,
                                    )
                                )
                                if diff < 8:  # <12.5% difference = same frame
                                    skip_ocr = True
                                    if hasattr(self, "_last_ocr_text"):
                                        ocr_text = (
                                            self._last_ocr_text
                                        )  # Reuse previous result
                                        ocr_boxes = getattr(
                                            self, "_last_ocr_boxes", []
                                        )
                                    logger.debug(
                                        f"[OCR] Skipped unchanged frame (hash diff: {diff})"
                                    )

                            self._last_ocr_hash = current_hash
                        except Exception as e:
                            logger.debug(f"[OCR] Hash comparison failed: {e}")

                    # Gate: Only run OCR if frame likely contains text (edge density check)
                    # Check enable_ocr master switch
                    if getattr(settings, "enable_ocr", True):
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
                                self._last_ocr_time = (
                                    time.time()
                                )  # Update last run time
                                logger.info(
                                    f"[OCR] Extracted: {ocr_text[:100]}..."
                                )
                            else:
                                logger.debug(
                                    "[OCR] No text found in gated frame"
                                )
                    else:
                        logger.debug("[OCR] Disabled via config")

                    # Deep Research Enrichment (REMOVED: Content Moderation & Clock Reader as per user request)
                    pass

            except Exception as e:
                logger.warning(f"[OCR] Failed: {e}")

            # ============================================================
            # OBJECT DETECTION: YOLO-World for general objects (lazy-loaded)
            # ============================================================
            detected_objects: list[str] = []
            try:
                if (
                    getattr(settings, "enable_object_detection", True)
                    and self.enhanced_config
                    and self.enhanced_config.object_detector
                ):
                    obj_detector = self.enhanced_config.object_detector
                    if frame_img is not None:
                        # Run YOLO-World detection on frame
                        detections = obj_detector.detect(frame_img)
                        if detections:
                            detected_objects = [
                                d.get("label", d.get("class", ""))
                                for d in detections
                                if d.get("confidence", 0) > 0.3
                            ]
                            if detected_objects:
                                unique_objects = list(set(detected_objects))
                                video_context += f"\n[DETECTED-OBJECTS]: {', '.join(unique_objects)}"
                                logger.debug(
                                    f"[ObjectDetection] Found: {unique_objects}"
                                )
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
                payload["visible_text"] = analysis.scene.visible_text if analysis.scene else []
                # CRITICAL FIX: Merge OCR-detected text with VLM-detected text
                if ocr_text:
                    # Split OCR text into searchable tokens
                    ocr_tokens = [w.strip() for w in ocr_text.split() if len(w.strip()) > 2]
                    # Extend both tokenized words and full text
                    current_text = payload["visible_text"]
                    if isinstance(current_text, list):
                        current_text.extend(ocr_tokens)
                        if len(ocr_text) > 3:
                            current_text.append(ocr_text.strip())
                        payload["visible_text"] = list(set(current_text))
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
                # NEW: Store YOLO-detected objects as dedicated searchable field
                # Enables queries like "frames with cars" via object_labels text index
                if detected_objects:
                    payload["object_labels"] = list(set(detected_objects))
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

            # ============================================================
            # DOMINANT COLOR EXTRACTION: Enables "blue scenes" queries
            # Uses lightweight k-means on downsampled frame for efficiency
            # ============================================================
            try:
                if frame_img is not None:
                    
                    small_frame = cv2.resize(frame_img, (32, 32))
                    pixels = small_frame.reshape(-1, 3).astype(np.float32)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
                    
                    counts = np.bincount(labels.flatten())
                    dominant_idx = np.argmax(counts)
                    dominant_bgr = centers[dominant_idx].astype(int)
                    
                    r, g, b = int(dominant_bgr[2]), int(dominant_bgr[1]), int(dominant_bgr[0])
                    r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
                    max_c, min_c = max(r_n, g_n, b_n), min(r_n, g_n, b_n)
                    delta = max_c - min_c
                    
                    val = max_c * 100
                    sat = (delta / max_c * 100) if max_c > 0 else 0
                    hue = 0
                    if delta > 0:
                        if max_c == r_n:
                            hue = 60 * (((g_n - b_n) / delta) % 6)
                        elif max_c == g_n:
                            hue = 60 * (((b_n - r_n) / delta) + 2)
                        else:
                            hue = 60 * (((r_n - g_n) / delta) + 4)
                    
                    payload["dominant_color_rgb"] = [r, g, b]
                    payload["dominant_color_hsv"] = [round(hue, 1), round(sat, 1), round(val, 1)]
            except Exception as e:
                logger.debug(f"[DominantColor] Skipped: {e}")

            # 3d. Add structured face data for UI overlays (bboxes)
            # This enables drawing boxes around identified people in the UI
            faces_metadata = []
            for face, cluster_id in zip(
                detected_faces, face_cluster_ids, strict=False
            ):
                face_name = self.db.get_face_name_by_cluster(cluster_id)
                bbox = (
                    face.bbox
                    if isinstance(face.bbox, list)
                    else list(face.bbox)
                )
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

