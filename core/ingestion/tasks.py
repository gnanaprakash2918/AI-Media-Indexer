import asyncio
from pathlib import Path

from celery import shared_task
from celery.utils.log import get_task_logger

from core.ingestion.jobs import JobStatus, job_manager
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import bind_context, clear_context

logger = get_task_logger(__name__)


@shared_task(bind=True, name="core.ingestion.tasks.ingest_video_task")
def ingest_video_task(self, video_path: str, job_id: str):
    """Celery task to ingest a video file."""
    bind_context(trace_id=job_id, component="celery_worker")
    logger.info(f"Starting distributed ingestion for {video_path} (Job {job_id})")

    # We need a dedicated pipeline instance for this worker process
    # Since Celery workers fork, we should initialize it here or use a global lazy one.
    # Safe bet: Initialize new pipeline instance per task or per worker.
    # Instantiating per task is safer to avoid state leak.

    try:
        # Update Job Status (if not already handled by caller)
        # Actually caller creates job. We just update progress?
        # Pipeline updates progress natively if configured correctly.

        pipeline = IngestionPipeline()

        # Run synchronous or run async via run_until_complete?
        # ingest_video is async? No, pipeline methods are often async but `process_video`?
        # `pipeline.process_video` is async.

        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # We assume pipeline has a way to process by path.
        # But wait, `pipeline.py` currently has `start_ingest` which spawns background task.
        # We want to run it BLOCKING here in the worker.

        # We need a blocking entry point in Pipeline.
        # `process_video` is the internal method.
        # `run_ingest` wrapper?

        # Let's peek at pipeline usage.
        # It calls `pipeline.start_ingest(media_file)`

        # We'll use asyncio.run to execute the async process_video logic.

        async def _run():
            await pipeline.process_video(Path(video_path), job_id=job_id)

        loop.run_until_complete(_run())

        logger.info(f"Finished ingestion for {video_path}")
        return {"status": "completed", "path": video_path}

    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}")
        # Job status 'failed' should be set by pipeline error handling
        # But if it crashes hard, we might need to set it here.
        job_manager.update_job(job_id, status=JobStatus.FAILED, error=str(exc))
        raise self.retry(exc=exc, countdown=60, max_retries=3)
    finally:
        clear_context()
