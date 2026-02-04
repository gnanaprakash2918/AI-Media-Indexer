"""Server-Sent Events (SSE) for real-time job updates.

Exposes the existing progress_tracker.event_stream() to the API
for frontend consumption with reconnection support.
"""

from __future__ import annotations

from fastapi import APIRouter, Header
from sse_starlette.sse import EventSourceResponse

from core.utils.logger import get_logger
from core.utils.progress import progress_tracker

log = get_logger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


@router.get("")
async def stream_all_events(
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
):
    """Stream all job progress updates via SSE.

    Supports reconnection via Last-Event-ID header - will replay
    any missed events since that ID.

    Events format:
        {event, event_id, job_id, status, progress, stage, message, ...}

    Event types:
        - job_started: New job began processing
        - job_progress: Progress update
        - stage_start: Pipeline stage started
        - stage_complete: Pipeline stage finished
        - job_granular_update: Detailed frame/timestamp update
        - job_completed: Job finished successfully
        - job_failed: Job failed with error
        - heartbeat: Keep-alive ping (every 15s)
    """
    last_id = int(last_event_id) if last_event_id else None
    if last_id:
        log.info(f"[SSE] Client reconnecting from event ID {last_id}")
    else:
        log.info("[SSE] Client connected for job events")

    return EventSourceResponse(
        progress_tracker.event_stream(last_event_id=last_id),
        media_type="text/event-stream",
    )


@router.get("/jobs")
async def stream_job_progress(
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
):
    """Stream all job progress updates via SSE (legacy endpoint).

    Kept for backward compatibility. Use /events for new integrations.
    """
    last_id = int(last_event_id) if last_event_id else None
    return EventSourceResponse(
        progress_tracker.event_stream(last_event_id=last_id),
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
    stats = progress_tracker.get_job_stats(job_id)
    if not stats:
        return {"error": f"Job {job_id} not found"}
    return stats
