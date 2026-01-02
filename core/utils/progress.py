"""Global progress tracking for ingestion jobs with SSE support."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, AsyncGenerator


class JobStatus(str, Enum):
    """Status of a processing job."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Information about a processing job."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    file_path: str = ""
    media_type: str = "unknown"
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    error: str | None = None
    current_stage: str = ""
    
    # Granular Progress Details
    total_frames: int = 0
    processed_frames: int = 0
    current_frame_timestamp: float = 0.0
    total_duration: float = 0.0


class ProgressTracker:
    """Thread-safe progress tracking with SSE broadcasting."""

    def __init__(self) -> None:
        """Initialize the progress tracker."""
        self._lock = Lock()
        self._jobs: dict[str, JobInfo] = {}
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def start(
        self,
        job_id: str,
        file_path: str = "",
        media_type: str = "unknown",
    ) -> None:
        """Start tracking a job."""
        with self._lock:
            self._jobs[job_id] = JobInfo(
                job_id=job_id,
                status=JobStatus.RUNNING,
                file_path=file_path,
                media_type=media_type,
            )
        self._broadcast({
            "event": "job_started",
            "job_id": job_id,
            "file_path": file_path,
            "media_type": media_type,
        })

    def update(
        self,
        job_id: str,
        percent: float,
        stage: str = "",
        message: str = "",
    ) -> None:
        """Update job progress (0-100)."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.progress = min(100.0, max(0.0, percent))
                if stage:
                    job.current_stage = stage
                if message:
                    job.message = message
        self._broadcast({
            "event": "job_progress",
            "job_id": job_id,
            "progress": percent,
            "stage": stage,
            "message": message,
        })

    def complete(self, job_id: str, message: str = "Processing complete") -> None:
        """Mark job as complete."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.completed_at = time.time()
                job.message = message
        self._broadcast({
            "event": "job_completed",
            "job_id": job_id,
            "message": message,
        })

    def fail(self, job_id: str, error: str = "Unknown error") -> None:
        """Mark job as failed."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = JobStatus.FAILED
                job.progress = -1.0
                job.completed_at = time.time()
                job.error = error
        self._broadcast({
            "event": "job_failed",
            "job_id": job_id,
            "error": error,
        })

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job. Returns True if cancelled."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job.status in (JobStatus.RUNNING, JobStatus.PAUSED):
                    job.status = JobStatus.CANCELLED
                    job.completed_at = time.time()
                    self._broadcast({
                        "event": "job_cancelled",
                        "job_id": job_id,
                    })
                    return True
        return False

    def pause(self, job_id: str) -> bool:
        """Pause a running job."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.PAUSED
                    self._broadcast({"event": "job_paused", "job_id": job_id})
                    return True
        return False

    def resume(self, job_id: str) -> bool:
        """Resume a paused job."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job.status == JobStatus.PAUSED:
                    job.status = JobStatus.RUNNING
                    self._broadcast({"event": "job_resumed", "job_id": job_id})
                    return True
        return False

    def update_granular(
        self,
        job_id: str,
        processed_frames: int | None = None,
        total_frames: int | None = None,
        current_timestamp: float | None = None,
        total_duration: float | None = None,
    ) -> None:
        """Update granular progress details."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if processed_frames is not None:
                    job.processed_frames = processed_frames
                if total_frames is not None:
                    job.total_frames = total_frames
                if current_timestamp is not None:
                    job.current_frame_timestamp = current_timestamp
                if total_duration is not None:
                    job.total_duration = total_duration
        
        # Broadcast detailed update (could throttle this if needed)
        self._broadcast({
            "event": "job_granular_update",
            "job_id": job_id,
            "processed_frames": job.processed_frames,
            "total_frames": job.total_frames,
            "timestamp": job.current_frame_timestamp,
            "duration": job.total_duration
        })

    def is_paused(self, job_id: str) -> bool:
        """Check if a job is paused."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job is not None and job.status == JobStatus.PAUSED

    def get(self, job_id: str) -> JobInfo | None:
        """Get job info by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_all(self) -> list[JobInfo]:
        """Get all jobs."""
        with self._lock:
            return list(self._jobs.values())

    def is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job is not None and job.status == JobStatus.CANCELLED

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to job events."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Unsubscribe from job events."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def _broadcast(self, event: dict[str, Any]) -> None:
        """Broadcast event to all subscribers."""
        event["timestamp"] = time.time()
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def event_stream(self) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator for SSE events."""
        queue = self.subscribe()
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self.unsubscribe(queue)


progress_tracker = ProgressTracker()
