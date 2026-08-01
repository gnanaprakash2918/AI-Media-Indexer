"""MediaIngestDispatcher — entry point for the parallel ingestion DAG.

The dispatcher replaces the serial pipeline.ingest() call for long media.
For short media (< min_media_length_for_chunking), it falls back to the
original serial IngestionPipeline path to avoid overhead.

Responsibilities:
  1. Probe the media file (duration, sha256, codec).
  2. Compute chunk boundaries (aligned to chunk_duration_seconds).
  3. For each chunk: compute chunk_id, insert chunk_state rows for all
     applicable stages (idempotent — safe to call on restart).
  4. Set the Redis fan-in expected count for each chunk.
  5. Enqueue 4 parallel Celery tasks per chunk (scene, speech,
     audio_event [if enabled], metadata).

Idempotency:
  The dispatcher can be called multiple times for the same media_id without
  creating duplicate work. chunk_state upserts are ON CONFLICT DO UPDATE;
  chunks with status='completed' are skipped at the worker level.

Usage:
    from core.ingestion.dispatcher import MediaIngestDispatcher

    dispatcher = MediaIngestDispatcher()
    job_id = await dispatcher.dispatch("/path/to/video.mp4")
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from pathlib import Path

from config import settings
from core.domain.chunk import (
    FANIN_BASE_COUNT,
    ChunkStage,
    ChunkStatus,
    compute_chunk_boundaries,
    make_chunk_id,
)
from core.ingestion.workers.fanin import set_fanin_expected
from core.storage.repositories.chunk_state_repo import ChunkStateRepository
from core.utils.logger import logger


class MediaIngestDispatcher:
    """Dispatch a media file to the parallel ingestion DAG.

    For media shorter than min_media_length_for_chunking (default 30 min),
    falls back to the serial IngestionPipeline to avoid queue/DB overhead.
    """

    def __init__(
        self,
        repo: ChunkStateRepository | None = None,
        *,
        chunk_duration_s: float | None = None,
        min_length_for_chunking_s: float | None = None,
        stage_version: str | None = None,
    ) -> None:
        self._repo = repo or ChunkStateRepository.from_settings()
        self._chunk_duration_s = chunk_duration_s or settings.chunk_duration_seconds
        self._min_length_s = (
            min_length_for_chunking_s or settings.min_media_length_for_chunking
        )
        self._stage_version = stage_version or settings.ingestion_stage_version

    async def dispatch(self, media_path: str | Path) -> str:
        """Dispatch a media file to the parallel ingestion DAG.

        Args:
            media_path: Absolute path to the media file.

        Returns:
            job_id: Unique identifier for this ingestion job.
        """
        path = Path(media_path)
        job_id = str(uuid.uuid4())

        logger.info(f"[Dispatcher] Starting dispatch for {path.name} job={job_id}")

        # 1. Probe the media (duration + sha256)
        duration_s, source_sha256 = await self._probe(path)

        # 2. Decide: chunk or serial
        if duration_s < self._min_length_s:
            logger.info(
                f"[Dispatcher] {path.name} is {duration_s:.0f}s "
                f"< {self._min_length_s:.0f}s threshold — using serial pipeline."
            )
            return await self._dispatch_serial(path, job_id)

        # 3. Compute stable media_id from file hash
        media_id = source_sha256[:32]  # first 32 chars of sha256 — short enough for logs

        logger.info(
            f"[Dispatcher] {path.name} is {duration_s / 60:.1f}min — "
            f"dispatching to parallel DAG with {self._chunk_duration_s / 60:.0f}min chunks."
        )

        # 4. Compute chunk boundaries
        boundaries = compute_chunk_boundaries(duration_s, self._chunk_duration_s)
        logger.info(
            f"[Dispatcher] {len(boundaries)} chunks for media_id={media_id[:8]}"
        )

        # 5. Determine fan-in expected count (once, shared across all chunks)
        expected_tracks = self._compute_expected_track_count()

        # 6. For each chunk: upsert state rows + set fan-in + enqueue tasks
        for chunk_index, (chunk_start_s, chunk_end_s) in enumerate(boundaries):
            chunk_id = make_chunk_id(media_id, chunk_index, source_sha256)

            await self._init_chunk_state(
                chunk_id=chunk_id,
                media_id=media_id,
                chunk_index=chunk_index,
            )

            # Set Redis fan-in expected count (idempotent: overwrite is safe)
            set_fanin_expected(chunk_id, expected_tracks)

            # Enqueue parallel tracks
            self._enqueue_tracks(
                chunk_id=chunk_id,
                media_id=media_id,
                chunk_index=chunk_index,
                media_path=str(path),
                chunk_start_s=chunk_start_s,
                chunk_end_s=chunk_end_s,
                stage_version=self._stage_version,
                job_id=job_id,
            )

            logger.debug(
                f"[Dispatcher] Enqueued chunk {chunk_index + 1}/{len(boundaries)} "
                f"chunk_id={chunk_id[:8]} "
                f"({chunk_start_s / 60:.1f}m–{chunk_end_s / 60:.1f}m)"
            )

        logger.info(
            f"[Dispatcher] Dispatched {len(boundaries)} chunks × "
            f"{expected_tracks} tracks for job={job_id}"
        )
        return job_id

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    async def _probe(self, path: Path) -> tuple[float, str]:
        """Return (duration_seconds, sha256_hex) for the media file.

        sha256 is computed from the file bytes — ensures that re-uploading
        a different file with the same name produces a new chunk_id.
        """
        try:
            from core.processing.prober import MediaProber

            prober = MediaProber()
            info = prober.probe(str(path))
            duration_s = float(info.get("duration", 0))
        except Exception as e:
            logger.warning(
                f"[Dispatcher] MediaProber failed ({e}), falling back to ffprobe."
            )
            duration_s = await self._ffprobe_duration(path)

        source_sha256 = await asyncio.get_event_loop().run_in_executor(
            None, self._compute_sha256, path
        )
        return duration_s, source_sha256

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    async def _ffprobe_duration(path: Path) -> float:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        try:
            return float(stdout.decode().strip())
        except ValueError:
            return 0.0

    async def _init_chunk_state(
        self,
        chunk_id: str,
        media_id: str,
        chunk_index: int,
    ) -> None:
        """Upsert chunk_state rows for all applicable stages (idempotent)."""
        stages = [
            ChunkStage.SCENE,
            ChunkStage.SPEECH,
            ChunkStage.METADATA,
            ChunkStage.FUSION,
        ]
        if settings.enable_audio_events:
            stages.append(ChunkStage.AUDIO_EVENT)

        for stage in stages:
            await self._repo.upsert_state(
                chunk_id=chunk_id,
                media_id=media_id,
                chunk_index=chunk_index,
                stage=stage,
                stage_version=self._stage_version,
                status=ChunkStatus.PENDING,
            )

    def _compute_expected_track_count(self) -> int:
        """Return the number of tracks that must INCR fan-in before Fusion fires.

        Base: 2 (scene-chain + speech)
        +1 if enable_audio_events
        +1 for metadata

        Why scene+ocr_vision count as 1:
          SceneAgent chains directly to OCRAgent/VisionAgent. The pair
          contributes a single INCR to fan-in (OCR/Vision can't start
          before Scene, so they are not independent tracks).
        """
        count = FANIN_BASE_COUNT  # scene-chain=1 + speech=1
        if settings.enable_audio_events:
            count += 1
        count += 1  # metadata (always enabled)
        return count

    def _enqueue_tracks(
        self,
        *,
        chunk_id: str,
        media_id: str,
        chunk_index: int,
        media_path: str,
        chunk_start_s: float,
        chunk_end_s: float,
        stage_version: str,
        job_id: str,
    ) -> None:
        """Enqueue all 4 (or 3) parallel tracks for one chunk."""
        common = {
            "chunk_id": chunk_id,
            "media_id": media_id,
            "chunk_index": chunk_index,
            "media_path": media_path,
            "stage_version": stage_version,
            "job_id": job_id,
        }

        from core.ingestion.workers.scene_worker import task_run_scene
        from core.ingestion.workers.speech_worker import task_run_speech
        from core.ingestion.workers.metadata_worker import task_run_metadata

        # Track 1: Scene + OCR/Vision (chained within the task)
        task_run_scene.apply_async(
            kwargs={
                **common,
                "chunk_start_s": chunk_start_s,
                "chunk_end_s": chunk_end_s,
            },
            queue="ingest:scene",
        )

        # Track 2: Speech (Whisper + Pyannote) — parallel with scene
        task_run_speech.apply_async(
            kwargs=common,
            queue="ingest:speech",
        )

        # Track 3: Audio events (CLAP) — optional
        if settings.enable_audio_events:
            from core.ingestion.workers.audio_event_worker import task_run_audio_event

            task_run_audio_event.apply_async(
                kwargs=common,
                queue="ingest:audio_event",
            )

        # Track 4: Metadata + subtitle parsing — parallel with all others
        task_run_metadata.apply_async(
            kwargs=common,
            queue="ingest:metadata",
        )

    async def _dispatch_serial(self, path: Path, job_id: str) -> str:
        """Fall back to the serial IngestionPipeline for short media."""
        from core.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        result_job_id = await pipeline.ingest(str(path))
        return result_job_id or job_id
