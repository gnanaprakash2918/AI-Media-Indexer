"""Scene worker — Track 1 of the parallel ingestion DAG.

Responsibilities:
  1. Run SceneAgent (shot boundary detection + adaptive keyframe extraction).
  2. On success: chain to OCR/Vision worker (same track — OCR/Vision cannot
     start before keyframes exist). Scene and OCR/Vision together count as
     ONE fan-in contribution.
  3. On failure: retry up to max_attempts, then quarantine and increment
     the fan-in counter so FusionAgent is not blocked.

Why scene and ocr_vision are one chained task (not two independent tasks):
  OCRAgent and VisionAgent require the keyframes produced by SceneAgent.
  Running them as separate fan-in tracks would require Scene to complete
  before the dispatcher even knows how many tracks to expect. Instead, Scene
  chains to OCR/Vision inline, and the pair contributes ONE INCR to fan-in.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.domain.chunk import ChunkStage
from core.ingestion.celery_app import celery_app
from core.ingestion.workers.fanin import increment_and_check
from core.utils.logger import logger


def _get_repo():
    from core.storage.repositories.chunk_state_repo import ChunkStateRepository
    return ChunkStateRepository.from_settings()


@celery_app.task(
    name="ingest.scene",
    bind=True,
    queue="ingest:scene",
    max_retries=None,  # retry logic handled manually below
    acks_late=True,
)
def task_run_scene(
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
    """Run scene detection and keyframe extraction for one chunk."""
    repo = _get_repo()

    async def _run() -> None:
        # Idempotency: skip if already done
        if await repo.is_completed(chunk_id, ChunkStage.SCENE):
            logger.info(
                f"[SceneWorker] chunk={chunk_id[:8]} already completed, skipping."
            )
            return

        await repo.mark_processing(chunk_id, ChunkStage.SCENE, stage_version)
        try:
            # Import pipeline lazily to avoid circular imports at module load
            from core.ingestion.pipeline import IngestionPipeline

            pipeline = IngestionPipeline()
            path = Path(media_path)

            # Run scene detection stage (produces keyframes + scene boundaries)
            await pipeline._process_frames(
                path,
                job_id,
                total_duration=chunk_end_s,
                chunk_start=chunk_start_s,
                chunk_end=chunk_end_s,
            )
            await pipeline._process_scene_captions(
                path,
                job_id,
                chunk_start=chunk_start_s,
                chunk_end=chunk_end_s,
            )

            await repo.mark_completed(chunk_id, ChunkStage.SCENE)
            logger.info(
                f"[SceneWorker] chunk={chunk_id[:8]} scene+ocr_vision COMPLETED"
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            await repo.mark_failed(chunk_id, ChunkStage.SCENE, error_msg)

            if await repo.should_quarantine(chunk_id, ChunkStage.SCENE):
                await repo.mark_quarantined(chunk_id, ChunkStage.SCENE, error_msg)
                logger.error(
                    f"[SceneWorker] chunk={chunk_id[:8]} QUARANTINED after "
                    f"exhausting retries: {error_msg}"
                )
                # Still increment fan-in — Fusion must not be blocked by a dead track
                if increment_and_check(chunk_id):
                    _trigger_fusion(chunk_id, media_id, chunk_index, stage_version, job_id)
                return  # DO NOT re-raise — poison-pill stop

            # Retriable: re-raise for Celery retry
            raise self.retry(exc=exc, countdown=30) from exc

        # Success path: increment fan-in
        if increment_and_check(chunk_id):
            _trigger_fusion(chunk_id, media_id, chunk_index, stage_version, job_id)

    asyncio.run(_run())


def _trigger_fusion(
    chunk_id: str,
    media_id: str,
    chunk_index: int,
    stage_version: str,
    job_id: str,
) -> None:
    """Enqueue the fusion task once all tracks have completed."""
    from core.ingestion.workers.fusion_worker import task_run_fusion

    logger.info(
        f"[FanIn] All tracks done for chunk={chunk_id[:8]}. Triggering fusion."
    )
    task_run_fusion.apply_async(
        kwargs={
            "chunk_id": chunk_id,
            "media_id": media_id,
            "chunk_index": chunk_index,
            "stage_version": stage_version,
            "job_id": job_id,
        },
        queue="ingest:fusion",
    )
