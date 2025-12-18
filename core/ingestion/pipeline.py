"""Media ingestion pipeline for multimodal indexing.

This module defines the `IngestionPipeline` class, which orchestrates the
end-to-end processing of a video file.
"""
from __future__ import annotations

import asyncio
import gc
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
from core.schemas import MediaMetadata, MediaType
from core.storage.db import VectorDB
from core.utils.logger import log
from core.utils.resource import resource_manager


class IngestionPipeline:
    """End-to-end media ingestion pipeline."""

    def __init__(
        self,
        *,
        qdrant_backend: str = settings.qdrant_backend,
        qdrant_host: str = settings.qdrant_host,
        qdrant_port: int = settings.qdrant_port,
        frame_interval_seconds: int = settings.frame_interval,
        tmdb_api_key: str | None = settings.tmdb_api_key,
    ) -> None:
        """Initialize the ingestion pipeline and its components.

        Args:
            qdrant_backend: Backend mode for Qdrant.
            qdrant_host: Hostname for Qdrant when using the "docker" backend.
            qdrant_port: TCP port for Qdrant when using the "docker" backend.
            frame_interval_seconds: Interval in seconds for frame extraction.
            tmdb_api_key: Optional TMDB API key for metadata enrichment.
        """
        log("[Pipeline] Initializing Infrastructure...")

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

        log("[Pipeline] Initialization complete. AI Models will load on-demand.")

    def _cleanup_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("[Pipeline] Memory cleanup triggered.")

    async def process_video(
        self,
        video_path: str | Path,
        media_type_hint: str = "unknown",
    ) -> None:
        """Process a single video file end-to-end.

        Args:
            video_path: Path to the input video file.
            media_type_hint: Optional string hint for the media type.
        """
        path = Path(video_path)

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Invalid media path: {path}")

        log(f"\n[Pipeline] Processing: {path.name}")

        hint_enum = (
            MediaType(media_type_hint)
            if media_type_hint in MediaType._value2member_map_
            else MediaType.UNKNOWN
        )

        meta: MediaMetadata = self.metadata_engine.identify(path, user_hint=hint_enum)
        log(
            f"[Metadata] {meta.media_type.value.upper()} | "
            f"{meta.title} ({meta.year})"
        )

        try:
            probed = self.prober.probe(path)
            duration = float(probed.get("format", {}).get("duration", 0.0))
            log(f"[Pipeline] Duration: {duration:.2f}s")
        except MediaProbeError as exc:
            log(f"[Pipeline] Probe failed: {exc}")
            raise

        log("[Pipeline] Step 1/3: Audio / Subtitles")

        audio_segments: list[dict[str, Any]] = []

        srt_path = path.with_suffix(".srt")
        if srt_path.exists():
            audio_segments = parse_srt(srt_path) or []

        if not audio_segments:
            try:
                await resource_manager.throttle_if_needed("compute")

                with AudioTranscriber() as transcriber:
                    temp_srt = path.with_suffix(".embedded.srt")
                    if transcriber._find_existing_subtitles(path, temp_srt, None, "ta"):
                        audio_segments = parse_srt(temp_srt) or []
                        if temp_srt.exists():
                            temp_srt.unlink()
            except Exception as exc:
                log(f"[Subtitle] Embedded extraction failed: {exc}")

        if not audio_segments:
            try:
                with AudioTranscriber() as transcriber:
                    audio_segments = transcriber.transcribe(path) or []
            except Exception as exc:
                log(f"[Whisper] Failed: {exc}")

        if audio_segments:
            prepared = self._prepare_segments_for_db(path=path, chunks=audio_segments)
            self.db.insert_media_segments(str(path), prepared)

        self._cleanup_memory()

        await resource_manager.throttle_if_needed("compute")
        log("[Pipeline] Step 2/3: Voice Analysis")

        self.voice = VoiceProcessor()
        try:
            voice_segments = await self.voice.process(path)
            if voice_segments:
                for seg in voice_segments:
                    try:
                        if seg.embedding is not None:
                            self.db.insert_voice_segment(
                                media_path=str(path),
                                start=seg.start_time,
                                end=seg.end_time,
                                speaker_label=seg.speaker_label,
                                embedding=seg.embedding,
                            )
                    except Exception as exc:
                        log(f"[Pipeline] Voice DB insert failed: {exc}")
        except Exception as exc:
            log(f"[Pipeline] Voice processing failed: {exc}")

        self._cleanup_memory()

        vision_task_type = "network" if settings.llm_provider == "gemini" else "compute"
        await resource_manager.throttle_if_needed(vision_task_type)

        log("[Pipeline] Step 3/3: Vision + Faces")

        self.vision = VisionAnalyzer()
        self.faces = FaceManager(use_gpu=settings.device == "cuda")

        frame_generator = self.extractor.extract(
            path,
            interval=self.frame_interval_seconds,
        )

        frame_count = 0

        try:
            async for frame_path in frame_generator:
                timestamp = frame_count * float(self.frame_interval_seconds)

                try:
                    await self._process_single_frame(
                        video_path=path,
                        frame_path=frame_path,
                        timestamp=timestamp,
                    )
                    await asyncio.sleep(1)
                except Exception as exc:
                    log(f"[Pipeline] Frame error: {exc}")
                finally:
                    if frame_path.exists():
                        try:
                            frame_path.unlink()
                        except Exception:
                            pass

                frame_count += 1

        finally:
            if self.vision:
                del self.vision
            if self.faces:
                del self.faces
            if self.voice:
                del self.voice

            self.vision = None
            self.faces = None
            self.voice = None

            self._cleanup_memory()

        log(f"[Pipeline] Finished {path.name}. Frames: {frame_count}")

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

    async def _process_single_frame(
        self,
        *,
        video_path: Path,
        frame_path: Path,
        timestamp: float,
    ) -> None:
        if not self.vision or not self.faces:
            return

        description: str | None = None

        try:
            description = await self.vision.describe(frame_path)
        except Exception:
            pass

        if description:
            try:
                vector = self.db.encoder.encode(description).tolist()
                self.db.upsert_media_frame(
                    point_id=f"{video_path}_{timestamp:.3f}",
                    vector=vector,
                    video_path=str(video_path),
                    timestamp=timestamp,
                    action=description,
                    dialogue=None,
                )
            except Exception as exc:
                log(f"[Pipeline] Vision DB insert failed: {exc}")

        try:
            detected_faces = await self.faces.detect_faces(frame_path)
        except Exception:
            return

        for face in detected_faces:
            if face.embedding is None:
                continue
            try:
                self.db.insert_face(face.embedding, name=None, cluster_id=None)
            except Exception as exc:
                log(f"[Pipeline] Face DB insert failed: {exc}")
