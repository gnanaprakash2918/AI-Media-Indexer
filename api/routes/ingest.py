"""API routes for media ingestion operations."""

import base64
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.deps import get_pipeline
from api.schemas import IngestRequest, ScanRequest
from config import settings
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import logger
from core.utils.observability import end_trace, start_trace
from core.utils.progress import progress_tracker

router = APIRouter()

# Media file validation
ALLOWED_MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".m4v",
    ".flv",  # Video
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",  # Audio
}


@router.post("/ingest")
async def ingest_media(
    ingest_request: IngestRequest,
    background_tasks: BackgroundTasks,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> dict:
    """Initiates the media ingestion pipeline for a specific file or URL.

    Validates the media type, resolves the provided path, and dispatches the
    processing job either to a local background task or a distributed Celery
    worker based on the system configuration.

    Args:
        ingest_request: Details of the media to ingest (path, type hints, bounds).
        background_tasks: FastAPI background task manager.
        pipeline: The core ingestion pipeline instance.

    Returns:
        A dictionary containing the generated 'job_id' and initial status.

    Raises:
        HTTPException: If the file is missing, the type is unsupported, or
            the pipeline is uninitialized.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if ingest_request.encoded_path:
        try:
            raw_path = base64.b64decode(ingest_request.encoded_path).decode(
                "utf-8"
            )
            logger.info(f"[Ingest] Base64 decoded: {raw_path!r}")
        except Exception as e:
            logger.warning(
                f"[Ingest] Base64 decode failed: {e}, falling back to path"
            )
            raw_path = ingest_request.path.strip().strip('"').strip("'")
    else:
        raw_path = ingest_request.path.strip().strip('"').strip("'")

    file_path = Path(raw_path).resolve()
    logger.info(f"[Ingest] Resolved: {file_path}, Exists: {file_path.exists()}")

    if not file_path.exists():
        decoded_path = Path(unquote(raw_path)).resolve()
        if decoded_path.exists():
            file_path = decoded_path
            logger.info(f"[Ingest] URL-decoded path worked: {file_path}")
        else:
            logger.warning(f"[Ingest] File not found. Raw: {raw_path!r}")
            raise HTTPException(
                status_code=404, detail=f"File not found: {raw_path}"
            )

    # Validate media file extension
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_MEDIA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_MEDIA_EXTENSIONS))}",
        )

    # Generate Job ID upfront so we can return it
    job_id = str(uuid4())

    # DISTRIBUTED INGESTION DISPATCH
    if settings.enable_distributed_ingestion:
        try:
            from core.ingestion.tasks import ingest_video_task

            # Initialize job tracking locally so UI sees it immediately
            progress_tracker.start(
                job_id,
                file_path=str(file_path),
                media_type=ingest_request.media_type_hint or "unknown",
                resume=False,
            )
            progress_tracker.update(
                job_id,
                0.0,
                stage="queued",
                message="Queued for distributed worker",
            )

            # Dispatch to Celery
            ingest_video_task.delay(str(file_path), job_id)  # type: ignore

            return {
                "status": "queued_distributed",
                "job_id": job_id,
                "file": str(file_path),
                "start_time": ingest_request.start_time,
                "message": "Distributed processing started.",
            }
        except Exception as e:
            logger.error(f"Failed to dispatch to Celery: {e}")
            raise HTTPException(
                status_code=500, detail=f"Distributed dispatch failed: {e}"
            ) from e

    async def run_pipeline():
        start_trace(
            name="background_ingest",
            metadata={"file": str(file_path), "job_id": job_id},
        )
        try:
            assert pipeline is not None
            await pipeline.process_video(
                file_path,
                ingest_request.media_type_hint,
                start_time=ingest_request.start_time,
                end_time=ingest_request.end_time,
                job_id=job_id,
                content_type_hint=ingest_request.content_type_hint,
            )
            end_trace("success")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            end_trace("error", str(e))

    background_tasks.add_task(run_pipeline)

    return {
        "status": "queued",
        "job_id": job_id,
        "file": str(file_path),
        "start_time": ingest_request.start_time,
        "end_time": ingest_request.end_time,
        "message": "Processing started. Use /events for live updates.",
    }


@router.post("/scan")
async def scan_library(
    request: ScanRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> dict:
    """Scans a local directory for new media files and reports discoveries.

    Filters files based on the allowed media extensions defined in the system.

    Args:
        request: The scan configuration (directory, optional extensions).
        pipeline: The core ingestion pipeline instance.

    Returns:
        A dictionary summary of found files and their absolute paths.

    Raises:
        HTTPException: If the pipeline is invalid or the scan fails.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline invalid")

    from core.ingestion.scanner import LibraryScanner

    scanner = LibraryScanner()
    new_files = []
    # Convert generator to list of paths
    for asset in scanner.scan(
        request.directory,
        excluded_dirs=None,  # Use defaults
    ):
        # Filter extensions if provided
        if request.extensions:
            # Check if file suffix is in requested extensions
            if asset.file_path.suffix.lower() in [
                e.lower() for e in request.extensions
            ]:
                new_files.append(asset.file_path)
        else:
            new_files.append(asset.file_path)

    return {
        "found": len(new_files),
        "files": [str(f) for f in new_files],
        "message": f"Found {len(new_files)} new files in {request.directory}",
    }


@router.get("/jobs")
async def list_jobs() -> dict:
    """Lists all historical and active processing jobs with progress details.

    Retrieves granular data from the global progress tracker, including
    stage messages, frame counts, and error details.

    Returns:
        A dictionary containing a list of job state records.
    """
    jobs = progress_tracker.get_all()
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "progress": j.progress,
                "file_path": j.file_path,
                "media_type": j.media_type,
                "current_stage": j.current_stage,
                "pipeline_stage": getattr(j, "pipeline_stage", "init"),
                "message": j.message,
                "started_at": j.started_at,
                "completed_at": j.completed_at,
                "error": j.error,
                # Granular stats
                "total_frames": j.total_frames,
                "processed_frames": j.processed_frames,
                "current_item_index": getattr(j, "current_item_index", 0),
                "total_items": getattr(j, "total_items", 0),
                "timestamp": j.current_frame_timestamp,
                "duration": j.total_duration,
                "last_heartbeat": getattr(j, "last_heartbeat", 0.0),
            }
            for j in jobs
        ]
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    """Retrieves full details and checkpoint data for a specific job.

    Args:
        job_id: The unique identifier of the job to retrieve.

    Returns:
        A dictionary containing the complete job state and progress statistics.

    Raises:
        HTTPException: If the job_id does not exist in the tracker.
        HTTPException: If the job_id does not exist in the tracker.
    """
    job = progress_tracker.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "file_path": job.file_path,
        "media_type": job.media_type,
        "current_stage": job.current_stage,
        "pipeline_stage": getattr(job, "pipeline_stage", "init"),
        "message": job.message,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error": job.error,
        # Granular stats
        "total_frames": job.total_frames,
        "processed_frames": job.processed_frames,
        "current_item_index": getattr(job, "current_item_index", 0),
        "total_items": getattr(job, "total_items", 0),
        "timestamp": job.current_frame_timestamp,
        "duration": job.total_duration,
        "last_heartbeat": getattr(job, "last_heartbeat", 0.0),
        "checkpoint_data": job.checkpoint_data,
    }


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    """Triggers an immediate cancellation of a running ingestion job.

    Attempts to notify the processing thread and clean up temporary assets.

    Args:
        job_id: The unique identifier of the job to cancel.

    Returns:
        A confirmation status dictionary.

    Raises:
        HTTPException: If the job cannot be found or is not in a cancellable state.
        HTTPException: If the job cannot be found or is not in a cancellable state.
    """
    success = progress_tracker.cancel(job_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Job not found or cannot be cancelled"
        )
    return {"status": "cancelled", "job_id": job_id}


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a running job."""
    success = progress_tracker.pause(job_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Job not found or cannot be paused"
        )
    return {"status": "paused", "job_id": job_id}


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused job."""
    success = progress_tracker.resume(job_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Job not found or cannot be resumed"
        )
    return {"status": "resumed", "job_id": job_id}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job from the system.

    Removes the job from both the progress tracker cache and the SQLite database.

    Args:
        job_id: The unique identifier of the job to delete.

    Returns:
        A confirmation status dictionary.

    Raises:
        HTTPException: If the job cannot be found.
    """
    success = progress_tracker.delete(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted", "job_id": job_id}
