"""Audio event worker — Track 3 (optional) of the parallel ingestion DAG.

Runs AudioEventAgent (CLAP model) to detect non-speech audio events:
music, applause, silence, environmental sounds.

This track is OPTIONAL. When settings.enable_audio_events = False:
  - The dispatcher does NOT enqueue this task.
  - The fan-in expected count is reduced by 1.
  - FusionAgent receives no audio event data and marks that modality absent.

Runs in parallel with scene and speech — reads only the audio stream, no
dependency on keyframes or transcription.
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
    name="ingest.audio_event",
    bind=True,
    queue="ingest:audio_event",
    max_retries=None,
    acks_late=True,
)
def task_run_audio_event(
    self,
    *,
    chunk_id: str,
    media_id: str,
    chunk_index: int,
    media_path: str,
    stage_version: str,
    job_id: str,
) -> None:
    """Run CLAP audio event detection for one chunk."""
    repo = _get_repo()

    async def _run() -> None:
        if await repo.is_completed(chunk_id, ChunkStage.AUDIO_EVENT):
            logger.info(
                f"[AudioEventWorker] chunk={chunk_id[:8]} already completed, skipping."
            )
            return

        await repo.mark_processing(chunk_id, ChunkStage.AUDIO_EVENT, stage_version)
        try:
            from core.ingestion.pipeline import IngestionPipeline

            pipeline = IngestionPipeline()
            path = Path(media_path)

            await pipeline._process_audio_events(path, job_id)

            await repo.mark_completed(chunk_id, ChunkStage.AUDIO_EVENT)
            logger.info(
                f"[AudioEventWorker] chunk={chunk_id[:8]} audio_event COMPLETED"
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            await repo.mark_failed(chunk_id, ChunkStage.AUDIO_EVENT, error_msg)

            if await repo.should_quarantine(chunk_id, ChunkStage.AUDIO_EVENT):
                await repo.mark_quarantined(chunk_id, ChunkStage.AUDIO_EVENT, error_msg)
                logger.error(
                    f"[AudioEventWorker] chunk={chunk_id[:8]} QUARANTINED: {error_msg}"
                )
                if increment_and_check(chunk_id):
                    _trigger_fusion(chunk_id, media_id, chunk_index, stage_version, job_id)
                return

            raise self.retry(exc=exc, countdown=30) from exc

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
