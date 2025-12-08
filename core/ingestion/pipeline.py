"""Media ingestion pipeline for multimodal indexing.

This module defines the `IngestionPipeline` class, which orchestrates the
end-to-end processing of a video file, including:

* Probing media metadata (duration, streams).
* Classifying media and enriching semantic metadata from filenames/APIs.
* Transcribing audio into text segments (with subtitle-first fallback).
* Extracting frames at regular intervals.
* Generating visual descriptions via a multimodal LLM.
* Detecting faces and generating embeddings.
* Storing all resulting vectors in Qdrant for retrieval.

Heavy components (Whisper, Vision LLM, Face Recognition) are "lazy-loaded"
on demand to manage VRAM usage on memory-constrained systems.

This module is used by the CLI entrypoint in `main.py`.
"""

from __future__ import annotations

import asyncio
import gc
from pathlib import Path
from typing import Any, Iterable

import torch

from core.processing.extractor import FrameExtractor
from core.processing.identity import FaceManager
from core.processing.metadata import MetadataEngine
from core.processing.prober import MediaProbeError, MediaProber
from core.processing.text_utils import parse_srt
from core.processing.transcriber import AudioTranscriber
from core.processing.vision import VisionAnalyzer
from core.schemas import MediaMetadata, MediaType
from core.storage.db import VectorDB
from core.utils.logger import log


class IngestionPipeline:
    """End-to-end media ingestion pipeline.

    This orchestrates:

      * Probing media metadata (duration, streams).
      * Media classification and semantic metadata enrichment.
      * Audio transcription / subtitle parsing.
      * Frame extraction at a fixed interval.
      * Visual description via multimodal LLM.
      * Face detection and vector indexing.
      * Vector storage in Qdrant via `VectorDB`.

    The pipeline is designed to be called with a single video file at a time,
    typically from the CLI entrypoint in ``main.py``.
    """

    def __init__(
        self,
        *,
        qdrant_backend: str = "docker",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        frame_interval_seconds: int = 5,
        tmdb_api_key: str | None = None,
    ) -> None:
        """Initialize the ingestion pipeline and its components.

        Args:
            qdrant_backend: Backend mode for Qdrant. Either ``"memory"`` or
                ``"docker"`` as expected by :class:`VectorDB`.
            qdrant_host: Hostname for Qdrant when using the ``"docker"``
                backend.
            qdrant_port: TCP port for Qdrant when using the ``"docker"``
                backend.
            frame_interval_seconds: Interval (in seconds) at which frames are
                extracted from the video timeline.
            tmdb_api_key: Optional TMDB API key used by :class:`MetadataEngine`
                for online metadata enrichment. If ``None``, the engine falls
                back to environment configuration or offline-only behavior.
        """
        log("[Pipeline] Initializing Infrastructure...")

        # Infrastructure components (lightweight)
        self.prober = MediaProber()
        self.extractor = FrameExtractor()
        self.db = VectorDB(
            backend=qdrant_backend,
            host=qdrant_host,
            port=qdrant_port,
        )
        self.metadata_engine = MetadataEngine(tmdb_api_key=tmdb_api_key)

        self.frame_interval_seconds = frame_interval_seconds

        # Lazy-loaded heavy models
        self.vision: VisionAnalyzer | None = None
        self.faces: FaceManager | None = None

        log("[Pipeline] Initialization complete. AI Models will load on-demand.")

    def _cleanup_memory(self) -> None:
        """Force Python and CUDA to release memory."""
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

        Steps:
          0. Classify media and enrich high-level metadata.
          1. Probe technical metadata (duration, streams).
          2. Build dialogue timeline via subtitles / Whisper:
               * Sidecar subtitles (.srt) if present.
               * Embedded subtitles (via ffmpeg-based check).
               * Whisper transcription as a final fallback.
          3. Index audio/dialogue segments into Qdrant.
          4. Extract frames at a fixed interval.
          5. For each frame:
               * Generate a visual description.
               * Index the description as a media frame vector.
               * Detect faces and index their encodings.

        Args:
            video_path: Path to the input video file.
            media_type_hint: Optional string hint for the media type. Must
                match one of :class:`MediaType` values (e.g. ``"video"``,
                ``"movie"``, ``"tv"``, ``"personal"``). If unknown or invalid,
                automatic classification is used.

        Raises:
            FileNotFoundError: If the input file does not exist or is not a file.
            MediaProbeError: If ffprobe fails to analyze the file.
        """
        path = Path(video_path)

        if not path.exists():
            msg = f"Video file not found: {path}"
            log(f"[Pipeline] {msg}")
            raise FileNotFoundError(msg)

        if not path.is_file():
            msg = f"Expected a file, got a directory or special path: {path}"
            log(f"[Pipeline] {msg}")
            raise FileNotFoundError(msg)

        log(f"\n[Pipeline]  Processing: {path.name} ")

        #  STEP 0: SEMANTIC METADATA & CLASSIFICATION
        # Convert string hint to Enum, with graceful fallback
        hint_enum = (
            MediaType(media_type_hint)
            if media_type_hint in MediaType._value2member_map_
            else MediaType.UNKNOWN
        )

        log("[Pipeline] Analyzing semantic metadata...")
        meta: MediaMetadata = self.metadata_engine.identify(path, user_hint=hint_enum)
        log(
            f"[Metadata] Identified: {meta.media_type.value.upper()} | "
            f"Title: {meta.title} ({meta.year})"
        )
        if meta.cast:
            log(f"[Metadata] Cast: {', '.join(meta.cast[:3])}")
        if not meta.is_processed:
            log("[Metadata] Online enrichment unavailable or skipped.")

        #  STEP 1: TECHNICAL PROBE
        try:
            probed_meta = self.prober.probe(path)
            duration_raw = probed_meta.get("format", {}).get("duration", 0)
            try:
                duration = float(duration_raw)
            except (TypeError, ValueError):
                duration = 0.0
            log(f"[Pipeline] Duration: {duration:.2f}s")
        except MediaProbeError as exc:
            log(f"[Pipeline] Probe failed: {exc}")
            raise

        #  STEP 2: AUDIO & SUBTITLES (Priority Chain)
        log("[Pipeline] Step 1/2: Processing Audio/Subtitles...")

        audio_segments: list[dict[str, Any]] = []

        # Priority 1: Sidecar (.srt)
        srt_path = path.with_suffix(".srt")
        if srt_path.exists():
            log(f"[Subtitle] Found sidecar file: {srt_path.name}")
            audio_segments = parse_srt(srt_path) or []

        # Priority 2: Embedded subtitles (via AudioTranscriber utility)
        if not audio_segments:
            try:
                with AudioTranscriber() as transcriber:
                    temp_srt = path.with_suffix(".embedded.srt")
                    # Default language chain: try Tamil ("ta")/English ("en") or
                    # whatever logic `_find_existing_subtitles` implements.
                    if transcriber._find_existing_subtitles(path, temp_srt, None, "ta"):
                        log("[Subtitle] Extracted embedded subtitles.")
                        audio_segments = parse_srt(temp_srt)
                        if temp_srt.exists():
                            temp_srt.unlink()
            except Exception as exc:  # noqa: BLE001
                log(f"[Subtitle] Embedded extraction check failed: {exc}")

        # Priority 3: Whisper transcription
        if not audio_segments:
            log("[Subtitle] No external subtitles found. Spinning up Whisper...")
            try:
                with AudioTranscriber() as transcriber:
                    audio_segments = transcriber.transcribe(path) or []

            except Exception as exc:  # noqa: BLE001
                log(f"[Pipeline] Whisper failed: {exc}")

        # Index transcription/subtitle chunks into Qdrant if available
        if audio_segments:
            prepared_segments = self._prepare_segments_for_db(
                path=path,
                chunks=audio_segments,
            )
            try:
                # TODO: Attach `meta` (Title/Year) to payloads for richer queries
                self.db.insert_media_segments(str(path), prepared_segments)
                log(
                    f"[Pipeline] Indexed {len(prepared_segments)} audio segments "
                    "into Qdrant.",
                )
            except Exception as exc:  # noqa: BLE001
                log(f"[Pipeline] Failed to index audio segments: {exc}")
        else:
            log("[Pipeline] No audio segments found or processed.")

        # Force cleanup after Whisper to ensure VRAM is clean for Vision
        self._cleanup_memory()

        #  STEP 3: VISUAL ANALYSIS (Vision + Faces)
        log("[Pipeline] Step 2/2: Analyzing frames (vision + faces)...")

        log("[Pipeline] Loading Vision & Identity models...")
        self.vision = VisionAnalyzer()
        # Force CPU for Dlib to avoid VRAM conflict with other workloads
        self.faces = FaceManager(use_gpu=False)

        frame_generator = self.extractor.extract(
            path,
            interval=self.frame_interval_seconds,
        )

        frame_count = 0
        try:
            async for frame_path in frame_generator:
                timestamp = frame_count * float(self.frame_interval_seconds)
                log(
                    f"[Pipeline] Processing frame #{frame_count} "
                    f"at ~{timestamp:.2f}s: {frame_path.name}",
                )

                try:
                    await self._process_single_frame(
                        video_path=path,
                        frame_path=frame_path,
                        timestamp=timestamp,
                    )
                    await asyncio.sleep(1)
                except Exception as exc:  # noqa: BLE001
                    log(f"[Pipeline] Error processing frame {frame_path}: {exc}")
                finally:
                    if frame_path.exists():
                        try:
                            frame_path.unlink()
                        except Exception:  # noqa: BLE001
                            # Non-fatal; continue.
                            pass

                frame_count += 1
        finally:
            log("[Pipeline] Finished visual processing. Unloading models...")
            if self.vision:
                del self.vision
            if self.faces:
                del self.faces
            self.vision = None
            self.faces = None
            self._cleanup_memory()

        log(
            f"\n[Pipeline]  Finished processing {path.name}. "
            f"Frames processed: {frame_count} ",
        )

    def _prepare_segments_for_db(
        self,
        *,
        path: Path,
        chunks: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize transcriber/subtitle chunks into the VectorDB format.

        The pipeline currently supports both:

          * SRT-derived chunks (with ``"start"``/``"end"`` keys).
          * Whisper-derived chunks (with a ``"timestamp"`` tuple).

        :class:`VectorDB.insert_media_segments` expects:
        ``{"text", "start", "end", "type"}``.

        Args:
            path: Path to the source media file.
            chunks: Raw chunk dictionaries emitted by the transcriber or
                subtitle parser.

        Returns:
            A list of normalized segment dictionaries, suitable for
            :meth:`VectorDB.insert_media_segments`.
        """
        prepared: list[dict[str, Any]] = []
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue

            # Handle both SRT (start/end keys) and Whisper (timestamp tuple)
            if "start" in chunk and "end" in chunk:
                start, end = chunk["start"], chunk["end"]
            else:
                timestamp = chunk.get("timestamp") or (None, None)
                start, end = timestamp

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
        """Process a single extracted video frame.

        This performs:

          * Vision description via the configured LLM.
          * Vector embedding + Qdrant indexing for the frame.
          * Face detection and indexing of face embeddings.

        Args:
            video_path: Path to the source video file.
            frame_path: Path to the extracted frame image.
            timestamp: Approximate timestamp (in seconds) of the frame
                within the video.

        Raises:
            FileNotFoundError: If the frame image cannot be found on disk.
        """
        if self.vision is None or self.faces is None:
            log("[Pipeline] Error: Models not initialized. Skipping frame.")
            return

        if not frame_path.exists():
            msg = f"Frame file not found: {frame_path}"
            log(f"[Pipeline] {msg}")
            raise FileNotFoundError(msg)

        description: str | None = None
        try:
            description = await self.vision.describe(frame_path)
        except Exception as exc:  # noqa: BLE001
            log(f"[Pipeline] Vision description failed: {exc}")

        if description:
            description = description.strip()
            log(f"[Vision] Output: {description[:100]}...")
            if description:
                try:
                    vector = self.db.encoder.encode(description).tolist()
                    unique_str = f"{video_path}_{timestamp:.3f}"
                    self.db.upsert_media_frame(
                        point_id=unique_str,
                        vector=vector,
                        video_path=str(video_path),
                        timestamp=timestamp,
                        action=description,
                        dialogue=None,
                    )
                except Exception as exc:  # noqa: BLE001
                    log(f"[Pipeline] Failed to index frame description: {exc}")

        try:
            detected_faces = await asyncio.to_thread(
                self.faces.detect_faces,
                frame_path,
            )
        except FileNotFoundError:
            log(
                f"[Pipeline] Frame disappeared before face detection: {frame_path}",
            )
            return
        except Exception as exc:  # noqa: BLE001
            log(f"[Pipeline] Face detection failed on {frame_path}: {exc}")
            return

        if not detected_faces:
            log(f"[Pipeline] No faces detected at {timestamp:.2f}s.")
            return

        log(
            f"[Pipeline] [{timestamp:.2f}s] Detected {len(detected_faces)} face(s).",
        )

        for face in detected_faces:
            try:
                self.db.insert_face(face.encoding, name=None, cluster_id=None)
            except Exception as exc:  # noqa: BLE001
                log(f"[Pipeline] Failed to index face embedding: {exc}")
