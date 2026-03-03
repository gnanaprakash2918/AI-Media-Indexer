"""Audio events detection stage.

Extracted from IngestionPipeline to reduce God class size.
IngestionPipeline inherits from AudioEventsStageMixin to compose these methods.
"""

from __future__ import annotations

import asyncio
import gc
import json
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from config import settings
from core.errors import IngestionError
from core.storage.db import VectorDB
from core.utils.logger import log_verbose, logger
from core.utils.progress import progress_tracker

if TYPE_CHECKING:
    from core.ingestion.pipeline import IngestionPipeline


class AudioEventsStageMixin:
    """Audio events detection stage."""

    # These will be available via IngestionPipeline inheritance
    db: VectorDB
    # Type stub for type checkers (allows accessing self.* in mixin)
    def __getattr__(self, name: str) -> Any: ...

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

            from core.processing.audio_events import get_audio_detector

            # Model-based AST Prediction enabled. No hardcoded lists.

            detector = get_audio_detector()

            # Get duration via cached probe (avoids redundant FFprobe calls)
            try:
                probe_data = await self.get_probe_data(path)
                duration = float(probe_data.get("format", {}).get("duration", 0))
            except Exception as e:
                logger.warning(
                    f"Probe failed, falling back to librosa for duration: {e}"
                )
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
            clap_window = settings.clap_window_seconds
            clap_stride = settings.clap_stride_seconds

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
                        message=f"Detecting audio events: chunk {chunk_idx + 1}/{total_chunks} ({chunk_start:.0f}s-{chunk_end:.0f}s)",
                    )

                # Stream only this chunk using FFmpeg (doesn't load full file)
                try:
                    audio_chunk = await self._extract_audio_segment(
                        path, chunk_start, chunk_end, sample_rate
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to extract audio chunk {chunk_idx}: {e}"
                    )
                    continue

                if audio_chunk is None or len(audio_chunk) == 0:
                    continue

                # Split chunk into CLAP windows
                samples_per_window = int(clap_window * sample_rate)
                stride_samples = int(clap_stride * sample_rate)

                clap_chunks = []
                for i in range(
                    0, len(audio_chunk) - samples_per_window + 1, stride_samples
                ):
                    window = audio_chunk[i : i + samples_per_window]
                    window_start = chunk_start + (i / sample_rate)
                    clap_chunks.append((window, window_start))

                if not clap_chunks:
                    continue

                # Batch process this chunk's windows
                try:
                    # Request embeddings for vector storage
                    # 1. Predict Events (AST) - Dynamic
                    chunk_events = await detector.predict_events_dynamic(
                        clap_chunks,
                        sample_rate=sample_rate,
                        top_k=2,
                        threshold=0.15,
                    )
                    # 2. Compute Embeddings (CLAP) - Vector Search
                    chunk_embeddings = await detector.get_embeddings_batch(
                        clap_chunks,
                        sample_rate=sample_rate,
                    )
                except Exception as e:
                    logger.warning(
                        f"Audio detection failed for chunk {chunk_idx}: {e}"
                    )
                    continue

                # Store events with deduplication
                for (_window_audio, window_start), events, embedding in zip(
                    clap_chunks, chunk_events, chunk_embeddings, strict=True
                ):
                    if not events:
                        continue

                    window_end = window_start + clap_window

                    for event in events:
                        # Deduplicate events in overlap region
                        if self._is_duplicate_event(
                            event,
                            window_start,
                            previous_events,
                            overlap_seconds,
                        ):
                            continue

                        # Use insert_audio_event which handles the schema correctly
                        self.db.insert_audio_event(
                            media_path=str(path),
                            event_type=event["event"],
                            start_time=window_start,
                            end_time=window_end,
                            confidence=event["confidence"],
                            clap_embedding=embedding,  # Store genuine 512-dim CLAP vector
                        )
                        events_stored += 1
                        previous_events.append(
                            {
                                "event": event["event"],
                                "start_time": window_start,
                            }
                        )

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

        try:
            # Check for audio stream first
            probe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(path),
            ]
            probe_res = subprocess.run(
                probe_cmd, capture_output=True, text=True
            )
            if not probe_res.stdout.strip():
                return None

            cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                str(start),
                "-t",
                str(end - start),
                "-i",
                str(path),
                "-ar",
                str(sample_rate),  # Target sample rate
                "-f",
                "f32le",  # 32-bit float PCM
                "-",  # Output to stdout
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=60)

            if result.returncode != 0:
                logger.warning(
                    f"FFmpeg extraction failed: {result.stderr.decode()[:200]}"
                )
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
        self,
        event: dict,
        event_time: float,
        previous_events: list,
        overlap_window: float,
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
            if (
                prev["event"] == event["event"]
                and abs(prev["start_time"] - event_time) < overlap_window
            ):
                return True
        return False

