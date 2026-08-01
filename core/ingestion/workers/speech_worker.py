"""Speech worker — Track 2 of the parallel ingestion DAG.

Runs SpeechAgent (Whisper transcription) and VoiceAgent (Pyannote speaker
diarization) for one chunk. Independent of SceneAgent — runs on the audio
stream only and requires no keyframes.

SpeechAgent + VoiceAgent are bundled in one task because:
  - VoiceAgent (diarization) requires the audio stream produced by the same
    ffmpeg call that SpeechAgent uses.
  - Both are CPU/RAM-bound, not GPU-bound, so they can share a single worker.
  - They count as ONE fan-in contribution.
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
    name="ingest.speech",
    bind=True,
    queue="ingest:speech",
    max_retries=None,
    acks_late=True,
)
def task_run_speech(
    self,
    *,
    chunk_id: str,
    media_id: str,
    chunk_index: int,
    media_path: str,
    stage_version: str,
    job_id: str,
) -> None:
    """Run Whisper transcription + Pyannote diarization for one chunk."""
    repo = _get_repo()

    async def _run() -> None:
        if await repo.is_completed(chunk_id, ChunkStage.SPEECH):
            logger.info(
                f"[SpeechWorker] chunk={chunk_id[:8]} already completed, skipping."
            )
            return

        await repo.mark_processing(chunk_id, ChunkStage.SPEECH, stage_version)
        try:
            from core.ingestion.pipeline import IngestionPipeline

            pipeline = IngestionPipeline()
            path = Path(media_path)

            # SpeechAgent: dense Whisper transcription with word timestamps
            await pipeline._process_audio(path)

            # VoiceAgent: Pyannote speaker diarization (who said what when)
            await pipeline._process_voice(path)

            await repo.mark_completed(chunk_id, ChunkStage.SPEECH)
            logger.info(
                f"[SpeechWorker] chunk={chunk_id[:8]} speech COMPLETED"
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            await repo.mark_failed(chunk_id, ChunkStage.SPEECH, error_msg)

            if await repo.should_quarantine(chunk_id, ChunkStage.SPEECH):
                await repo.mark_quarantined(chunk_id, ChunkStage.SPEECH, error_msg)
                logger.error(
                    f"[SpeechWorker] chunk={chunk_id[:8]} QUARANTINED: {error_msg}"
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
