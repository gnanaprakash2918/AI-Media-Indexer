"""Media ingestion pipeline for multimodal indexing.

This module defines the `IngestionPipeline` class, which orchestrates the
end-to-end processing of a video file, including:

* Probing media metadata (duration, streams).
* Transcribing audio into text segments.
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
from core.processing.prober import MediaProbeError, MediaProber
from core.processing.transcriber import AudioTranscriber
from core.processing.vision import VisionAnalyzer
from core.storage.db import VectorDB


class IngestionPipeline:
    """End-to-end media ingestion pipeline.

    This orchestrates:
      * Probing media metadata (duration, streams).
      * Audio transcription with Faster-Whisper.
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
        """
        print("[Pipeline] Initializing Infrastructure...")

        # Infrastructure components (Lightweight)
        self.prober = MediaProber()
        self.extractor = FrameExtractor()

        self.db = VectorDB(
            backend=qdrant_backend,
            host=qdrant_host,
            port=qdrant_port,
        )

        self.frame_interval_seconds = frame_interval_seconds

        # LAZY LOADING: We do not initialize heavy models here.
        # They are initialized in process_video to prevent VRAM conflict.
        self.vision: VisionAnalyzer | None = None
        self.faces: FaceManager | None = None

        print("[Pipeline] Initialization complete. AI Models will load on-demand.")

    def _cleanup_memory(self):
        """Forces Python and CUDA to release memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Pipeline] Memory cleanup triggered.")

    async def process_video(self, video_path: str | Path) -> None:
        """Process a single video file end-to-end.

        Steps:
          1. Probe metadata.
          2. Transcribe audio to text segments.
          3. Index audio segments into Qdrant.
          4. Extract frames at a fixed interval.
          5. For each frame:
              * Generate a visual description.
              * Index the description as a media frame vector.
              * Detect faces and index their encodings.

        Args:
            video_path: Path to the input video file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            MediaProbeError: If ffprobe fails to analyze the file.
        """
        path = Path(video_path)

        if not path.exists():
            msg = f"Video file not found: {path}"
            print(f"[Pipeline] {msg}")
            raise FileNotFoundError(msg)

        if not path.is_file():
            msg = f"Expected a file, got a directory or special path: {path}"
            print(f"[Pipeline] {msg}")
            raise FileNotFoundError(msg)

        print(f"\n[Pipeline]  Processing: {path.name} ")

        #  STEP 1: METADATA
        try:
            meta = self.prober.probe(path)
            duration_raw = meta.get("format", {}).get("duration", 0)
            try:
                duration = float(duration_raw)
            except (TypeError, ValueError):
                duration = 0.0

            print(f"[Pipeline] Duration: {duration:.2f}s")
        except MediaProbeError as exc:
            print(f"[Pipeline] Probe failed: {exc}")
            raise

        #  STEP 2: AUDIO (High VRAM - Whisper)
        print("[Pipeline] Step 1/2: Processing Audio...")

        audio_segments: list[dict[str, Any]] | None = None
        srt_path = path.with_suffix(".srt")

        if srt_path.exists():
            print(f"[Pipeline] Found sidecar subtitle: {srt_path.name}")
            from core.processing.text_utils import parse_srt

            audio_segments = parse_srt(srt_path)
        else:
            # 2. Only spin up Transcriber if no SRT found
            print("[Pipeline] No subtitle file found. Spinning up Whisper...")
            try:
                # Context manager loads model -> transcribes -> UNLOADS model
                with AudioTranscriber() as transcriber:
                    audio_segments = transcriber.transcribe(path)
            except Exception as e:
                print(f"[Pipeline] Transcription skipped/failed: {e}")

        # Index transcription chunks into Qdrant if available.
        if audio_segments:
            prepared_segments = self._prepare_segments_for_db(
                path=path,
                chunks=audio_segments,
            )
            try:
                self.db.insert_media_segments(str(path), prepared_segments)
                print(
                    f"[Pipeline] Indexed {len(prepared_segments)} audio segments "
                    "into Qdrant.",
                )
            except Exception as exc:
                print(f"[Pipeline] Failed to index audio segments: {exc}")
        else:
            print("[Pipeline] No audio segments returned (silence or error).")

        # Force cleanup after Whisper to ensure VRAM is clean for Vision
        self._cleanup_memory()

        #  STEP 3: VISUAL ANALYSIS (High VRAM - Vision LLM)
        print("[Pipeline] Step 2/2: Analyzing frames (vision + faces)...")

        # Initialize Workers LOCALLY only when needed
        print("[Pipeline] Loading Vision & Identity models...")
        self.vision = VisionAnalyzer()
        self.faces = FaceManager(use_gpu=False)

        frame_generator = self.extractor.extract(
            path,
            interval=self.frame_interval_seconds,
        )

        frame_count = 0
        try:
            async for frame_path in frame_generator:
                timestamp = frame_count * float(self.frame_interval_seconds)
                print(
                    f"[Pipeline] Processing frame #{frame_count} at ~{timestamp:.2f}s: "
                    f"{frame_path.name}",
                )

                try:
                    await self._process_single_frame(
                        video_path=path,
                        frame_path=frame_path,
                        timestamp=timestamp,
                    )
                    await asyncio.sleep(1)
                except Exception as exc:
                    print(f"[Pipeline] Error processing frame {frame_path}: {exc}")
                finally:
                    # Cleanup: delete frame after processing to save disk space.
                    if frame_path.exists():
                        try:
                            frame_path.unlink()
                        except Exception:
                            # Non-fatal; continue.
                            pass

                frame_count += 1
        finally:
            print("[Pipeline] Finished visual processing. Unloading models...")
            # Explicitly delete the objects to free resources
            if self.vision:
                del self.vision
            if self.faces:
                del self.faces
            self.vision = None
            self.faces = None
            self._cleanup_memory()

        print(
            f"\n[Pipeline]  Finished processing {path.name}. "
            f"Frames processed: {frame_count} ",
        )

    def _prepare_segments_for_db(
        self, *, path: Path, chunks: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize transcriber chunks into the format expected by VectorDB.

        The current :class:`AudioTranscriber` returns chunks with the keys
        ``"text"`` and ``"timestamp"`` (a ``(start, end)`` tuple).

        :class:`VectorDB.insert_media_segments` expects:
        ``{"text", "start", "end", "type"}``.

        This adapter converts the shape accordingly.

        Args:
            path: Path to the source media file.
            chunks: Raw chunk dictionaries emitted by the transcriber.

        Returns:
            A list of normalized segment dictionaries, suitable for
            :meth:`VectorDB.insert_media_segments`.
        """
        prepared: list[dict[str, Any]] = []
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue

            #  FIX: Handle both SRT (start/end keys) and Whisper (timestamp tuple)
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
        # Type Guard: Ensure models were loaded by process_video before usage.
        if self.vision is None or self.faces is None:
            print("[Pipeline] Error: Models not initialized. Skipping frame.")
            return

        if not frame_path.exists():
            msg = f"Frame file not found: {frame_path}"
            print(f"[Pipeline] {msg}")
            raise FileNotFoundError(msg)

        description: str | None = None
        try:
            description = await self.vision.describe(frame_path)
        except Exception as exc:
            print(f"[Pipeline] Vision description failed: {exc}")

        if description:
            description = description.strip()
            print(f"[Vision] Output: {description[:100]}...")
            if description:
                try:
                    # Use VectorDB's encoder directly for frame descriptions.
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
                except Exception as exc:
                    print(f"[Pipeline] Failed to index frame description: {exc}")

        try:
            # Dlib is blocking, so we run it in a thread.
            detected_faces = await asyncio.to_thread(
                self.faces.detect_faces, frame_path
            )
        except FileNotFoundError:
            print(
                f"[Pipeline] Frame disappeared before face detection: {frame_path}",
            )
            return
        except Exception as exc:
            print(f"[Pipeline] Face detection failed on {frame_path}: {exc}")
            return

        if not detected_faces:
            print(f"[Pipeline] No faces detected at {timestamp:.2f}s.")
            return

        print(
            f"[Pipeline] [{timestamp:.2f}s] Detected {len(detected_faces)} face(s).",
        )

        for face in detected_faces:
            try:
                self.db.insert_face(face.encoding, name=None, cluster_id=None)
            except Exception as exc:
                print(f"[Pipeline] Failed to index face embedding: {exc}")
