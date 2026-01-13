"""Server-Sent Events (SSE) for real-time job updates.

Exposes the existing progress_tracker.event_stream() to the API
for frontend consumption.
"""

from __future__ import annotations

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from core.utils.logger import get_logger
from core.utils.progress import progress_tracker

log = get_logger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


@router.get("/jobs")
async def stream_job_progress():
    """Stream all job progress updates via SSE.

    Returns:
        EventSourceResponse with job progress events.

    Events format:
        {job_id, status, progress, stage, message, ...}
    """
    log.info("[SSE] Client connected for job events")
    return EventSourceResponse(
        progress_tracker.event_stream(),
        media_type="text/event-stream",
    )


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """Get current status for a specific job.

    Args:
        job_id: The job ID to check.

    Returns:
        Job status dict or error.
    """
    job = progress_tracker.get(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}

    return {
        "job_id": job.job_id,
        "status": job.status.value if hasattr(job.status, "value") else job.status,
        "progress": job.progress,
        "stage": job.stage.value if hasattr(job.stage, "value") else str(job.stage),
        "message": job.message,
        "file_path": job.file_path,
    }
