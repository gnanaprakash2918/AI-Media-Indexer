"""Media ingestion pipeline orchestrator."""

from __future__ import annotations

import asyncio
import gc
import uuid
from pathlib import Path
from typing import Any, Iterable

import torch

from config import settings
from core.processing.extractor import FrameExtractor
from core.processing.identity import FaceManager
from core.processing.metadata import MetadataEngine
from core.processing.prober import MediaProbeError, MediaProber
from core.processing.text_utils import parse_srt
from core.processing.transcriber import AudioTranscriber
from core.processing.vision import VisionAnalyzer
from core.processing.voice import VoiceProcessor
from core.schemas import MediaType
from core.storage.db import VectorDB
from core.utils.frame_sampling import FrameSampler
from core.utils.logger import bind_context
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
        frame_interval_seconds: int = settings.frame_interval,
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

    def _cleanup_memory(self) -> None:
        """Force garbage collection and clear CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @observe("process_video")
    async def process_video(
        self,
        video_path: str | Path,
        media_type_hint: str = "unknown",
    ) -> str:
        """Process a single video file through the entire ingestion pipeline.

        Args:
            video_path: Path to the video file.
            media_type_hint: Optional hint about the media type (e.g., 'movie', 'eps').

        Returns:
            The job ID of the processing task.
        """
        job_id = str(uuid.uuid4())
        bind_context(component="pipeline")
        
        path = Path(video_path)
        progress_tracker.start(
            job_id,
            file_path=str(path),
            media_type=media_type_hint,
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
            _ = float(probed.get("format", {}).get("duration", 0.0))
        except MediaProbeError as e:
            progress_tracker.fail(job_id, error=f"Media probe failed: {e}")
            raise

        try:
            progress_tracker.update(job_id, 5.0, stage="audio", message="Processing audio")
            await retry(lambda: self._process_audio(path))
            progress_tracker.update(job_id, 30.0, stage="audio", message="Audio complete")

            if progress_tracker.is_cancelled(job_id):
                return job_id

            progress_tracker.update(job_id, 35.0, stage="voice", message="Processing voice")
            await retry(lambda: self._process_voice(path))
            progress_tracker.update(job_id, 50.0, stage="voice", message="Voice complete")

            if progress_tracker.is_cancelled(job_id):
                return job_id

            progress_tracker.update(job_id, 55.0, stage="frames", message="Processing frames")
            await retry(lambda: self._process_frames(path, job_id))
            progress_tracker.complete(job_id, message=f"Completed: {path.name}")

            return job_id

        except Exception as e:
            progress_tracker.fail(job_id, error=str(e))
            raise

    @observe("audio_processing")
    async def _process_audio(self, path: Path) -> None:
        audio_segments: list[dict[str, Any]] = []
        srt_path = path.with_suffix(".srt")

        if srt_path.exists():
            audio_segments = parse_srt(srt_path) or []

        if not audio_segments:
            await resource_manager.throttle_if_needed("compute")
            try:
                with AudioTranscriber() as transcriber:
                    temp_srt = path.with_suffix(".embedded.srt")
                    if transcriber._find_existing_subtitles(path, temp_srt, None, "ta"):
                        audio_segments = parse_srt(temp_srt) or []
                        if temp_srt.exists():
                            temp_srt.unlink()
            except Exception:
                pass

        if not audio_segments:
            await resource_manager.throttle_if_needed("compute")
            with AudioTranscriber() as transcriber:
                audio_segments = transcriber.transcribe(path) or []

        if audio_segments:
            prepared = self._prepare_segments_for_db(path=path, chunks=audio_segments)
            self.db.insert_media_segments(str(path), prepared)

        self._cleanup_memory()

    @observe("voice_processing")
    async def _process_voice(self, path: Path) -> None:
        await resource_manager.throttle_if_needed("compute")
        self.voice = VoiceProcessor()

        try:
            voice_segments = await self.voice.process(path)
            
            # Prepare voice thumbnails directory
            thumb_dir = settings.cache_dir / "thumbnails" / "voices"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            
            import subprocess
            import hashlib
            
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
    async def _process_frames(self, path: Path, job_id: str | None = None) -> None:
        vision_task_type = "network" if settings.llm_provider == "gemini" else "compute"
        await resource_manager.throttle_if_needed(vision_task_type)

        self.vision = VisionAnalyzer()
        self.faces = FaceManager(use_gpu=settings.device == "cuda")

        frame_generator = self.extractor.extract(
            path,
            interval=self.frame_interval_seconds,
        )

        frame_count = 0
        CLEANUP_INTERVAL = 10  # Cleanup memory every N frames

        async for frame_path in frame_generator:
            if job_id and progress_tracker.is_cancelled(job_id):
                break
            
            timestamp = frame_count * float(self.frame_interval_seconds)
            try:
                if self.frame_sampler.should_sample(frame_count):
                    await self._process_single_frame(
                        video_path=path,
                        frame_path=frame_path,
                        timestamp=timestamp,
                        index=frame_count,
                    )
                    if job_id and frame_count % 5 == 0:
                        progress = 55.0 + min(40.0, frame_count * 2.0)
                        progress_tracker.update(
                            job_id,
                            progress,
                            stage="frames",
                            message=f"Frame {frame_count}",
                        )
                await asyncio.sleep(0.3)  # Reduced sleep for faster processing
            finally:
                # Always delete the frame file after processing
                if frame_path.exists():
                    try:
                        frame_path.unlink()
                    except Exception:
                        pass
            
            frame_count += 1
            
            # Periodic memory cleanup to prevent memory buildup
            if frame_count % CLEANUP_INTERVAL == 0:
                self._cleanup_memory()

        # Final cleanup
        del self.vision
        del self.faces
        self.vision = None
        self.faces = None
        self._cleanup_memory()

    @observe("frame")
    async def _process_single_frame(
        self,
        *,
        video_path: Path,
        frame_path: Path,
        timestamp: float,
        index: int,
    ) -> None:
        if not self.vision or not self.faces:
            return

        description: str | None = None
        try:
            description = await self.vision.describe(frame_path)
        except Exception:
            pass

        if description:
            vector = self.db.encoder.encode(description).tolist()
            self.db.upsert_media_frame(
                point_id=f"{video_path}_{timestamp:.3f}",
                vector=vector,
                video_path=str(video_path),
                timestamp=timestamp,
                action=description,
                dialogue=None,
            )

        try:
            detected_faces = await self.faces.detect_faces(frame_path)
        except Exception:
            return

        # Save face thumbnails
        thumb_dir = settings.cache_dir / "thumbnails" / "faces"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a safe file prefix using hash of the filename
        import hashlib
        safe_stem = hashlib.md5(video_path.stem.encode()).hexdigest()

        for idx, face in enumerate(detected_faces):
            if face.embedding is not None:
                # Crop and save face thumbnail with better quality
                thumb_path: str | None = None
                try:
                    import cv2
                    import numpy as np
                    # Use cv2.imdecode for unicode path support on Windows
                    img_data = np.fromfile(str(frame_path), dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is not None:
                        # bbox format from FaceManager is (top, right, bottom, left)
                        top, right, bottom, left = face.bbox
                        face_w = right - left
                        face_h = bottom - top
                        # Increased padding for better context (40%)
                        pad = int(max(face_w, face_h) * 0.4)
                        y1 = max(0, top - pad)
                        y2 = min(img.shape[0], bottom + pad)
                        x1 = max(0, left - pad)
                        x2 = min(img.shape[1], right + pad)
                        face_crop = img[y1:y2, x1:x2]
                        
                        # Ensure minimum size for quality (resize up if too small)
                        min_size = 150
                        crop_h, crop_w = face_crop.shape[:2]
                        if crop_h > 0 and crop_w > 0:
                            if crop_h < min_size or crop_w < min_size:
                                scale = max(min_size / crop_h, min_size / crop_w)
                                new_w = int(crop_w * scale)
                                new_h = int(crop_h * scale)
                                face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                            
                            thumb_name = f"{safe_stem}_{timestamp:.2f}_{idx}.jpg"
                            thumb_file = thumb_dir / thumb_name
                            # Use higher JPEG quality (95 instead of default ~75)
                            cv2.imwrite(str(thumb_file), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            thumb_path = f"/thumbnails/faces/{thumb_name}"
                except Exception:
                    pass

                self.db.insert_face(
                    face.embedding,
                    name=None,
                    cluster_id=None,
                    media_path=str(video_path),
                    timestamp=timestamp,
                    thumbnail_path=thumb_path,
                )

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
