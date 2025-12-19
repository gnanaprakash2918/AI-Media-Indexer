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
        progress_tracker.start(job_id)

        path = Path(video_path)
        if not path.exists() or not path.is_file():
            progress_tracker.fail(job_id)
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
        except MediaProbeError:
            progress_tracker.fail(job_id)
            raise

        try:
            await retry(lambda: self._process_audio(path))
            progress_tracker.update(job_id, 30.0)

            await retry(lambda: self._process_voice(path))
            progress_tracker.update(job_id, 50.0)

            await retry(lambda: self._process_frames(path))
            progress_tracker.complete(job_id)

            return job_id

        except Exception:
            progress_tracker.fail(job_id)
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
            for seg in voice_segments or []:
                if seg.embedding is not None:
                    self.db.insert_voice_segment(
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        speaker_label=seg.speaker_label,
                        embedding=seg.embedding,
                    )
        finally:
            del self.voice
            self.voice = None
            self._cleanup_memory()

    @observe("frame_processing")
    async def _process_frames(self, path: Path) -> None:
        vision_task_type = "network" if settings.llm_provider == "gemini" else "compute"
        await resource_manager.throttle_if_needed(vision_task_type)

        self.vision = VisionAnalyzer()
        self.faces = FaceManager(use_gpu=settings.device == "cuda")

        frame_generator = self.extractor.extract(
            path,
            interval=self.frame_interval_seconds,
        )

        frame_count = 0

        async for frame_path in frame_generator:
            timestamp = frame_count * float(self.frame_interval_seconds)
            try:
                if self.frame_sampler.should_sample(frame_count):
                    await self._process_single_frame(
                        video_path=path,
                        frame_path=frame_path,
                        timestamp=timestamp,
                        index=frame_count,
                    )
                await asyncio.sleep(1)
            finally:
                if frame_path.exists():
                    try:
                        frame_path.unlink()
                    except Exception:
                        pass
            frame_count += 1

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

        for face in detected_faces:
            if face.embedding is not None:
                self.db.insert_face(face.embedding, name=None, cluster_id=None)

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
