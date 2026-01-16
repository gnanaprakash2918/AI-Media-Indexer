"""Global progress tracking for ingestion jobs with SSE support and persistence."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from threading import Lock
from typing import Any

from core.ingestion.jobs import JobInfo, JobStatus, PipelineStage, job_manager


class ProgressTracker:
    """Thread-safe progress tracking with SSE broadcasting and SQLite persistence."""

    def __init__(self) -> None:
        """Initialize the progress tracker."""
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        # In-memory cache for high-frequency polling/updates to avoid DB spam
        self._cache: dict[str, JobInfo] = {}
        self._last_db_update: dict[str, float] = {}
        self._lock = Lock()

        # Load active jobs from DB on startup
        self._sync_cache()

    def _sync_cache(self):
        """Load all jobs from DB to cache."""
        try:
            jobs = job_manager.get_all_jobs(limit=100)
            with self._lock:
                for job in jobs:
                    self._cache[job.job_id] = job
        except Exception:
            pass

    def start(
        self,
        job_id: str,
        file_path: str = "",
        media_type: str = "unknown",
        resume: bool = False,
    ) -> None:
        """Start tracking a job.

        Args:
            job_id: Unique identifier for the job.
            file_path: Path to the media file being processed.
            media_type: Type of media (video, audio, etc.).
            resume: If True, don't create a new DB entry, just update cache.
        """
        if resume:
            job = job_manager.get_job(job_id)
            if not job:
                # Fallback if inconsistent
                job = job_manager.create_job(job_id, file_path, media_type)
            else:
                job.status = JobStatus.RUNNING
                job_manager.update_job(job_id, status=JobStatus.RUNNING)
        else:
            # Create in DB
            job = job_manager.create_job(job_id, file_path, media_type)

        with self._lock:
            self._cache[job_id] = job

        self._broadcast(
            {
                "event": "job_started" if not resume else "job_resumed",
                "job_id": job_id,
                "file_path": file_path,
                "media_type": media_type,
            }
        )

    def update(
        self,
        job_id: str,
        percent: float,
        stage: str = "",
        message: str = "",
    ) -> None:
        """Update job progress (0-100)."""
        with self._lock:
            if job_id not in self._cache:
                # Try to load into cache if missing (e.g. after restart)
                job = job_manager.get_job(job_id)
                if job:
                    self._cache[job_id] = job
                else:
                    return

            job = self._cache[job_id]
            job.progress = min(100.0, max(0.0, percent))
            if stage:
                job.current_stage = stage
            if message:
                job.message = message

            # Broadcast immediately
            self._broadcast(
                {
                    "event": "job_progress",
                    "job_id": job_id,
                    "progress": job.progress,
                    "stage": job.current_stage,
                    "message": job.message,
                }
            )

            # Persist to DB (throttled)
            self._persist_throttled(job)

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
            if job_id not in self._cache:
                return
            job = self._cache[job_id]

            if processed_frames is not None:
                job.processed_frames = processed_frames
            if total_frames is not None:
                job.total_frames = total_frames
            if current_timestamp is not None:
                job.current_frame_timestamp = current_timestamp
            if total_duration is not None:
                job.total_duration = total_duration

            # Create checkpoint data for resuming (merge with existing)
            if current_timestamp is not None:
                existing_checkpoint = job.checkpoint_data or {}
                # Update only what changed, keep flags like "audio_complete"
                new_checkpoint = {
                    **existing_checkpoint,
                    "timestamp": current_timestamp,
                    "processed_frames": job.processed_frames,
                }
                job.checkpoint_data = new_checkpoint

            # Broadcast detailed update
            self._broadcast(
                {
                    "event": "job_granular_update",
                    "job_id": job_id,
                    "processed_frames": job.processed_frames,
                    "total_frames": job.total_frames,
                    "timestamp": job.current_frame_timestamp,
                    "duration": job.total_duration,
                }
            )

            # Persist to DB (throttled)
            self._persist_throttled(job)

    def _persist_throttled(self, job: JobInfo) -> None:
        """Persist job state to DB if enough time has passed."""
        import time as time_module

        now = time_module.time()
        last = self._last_db_update.get(job.job_id, 0)

        # Update DB every 2 seconds max for progress to reduce I/O
        if now - last > 2.0:
            try:
                job_manager.update_job(
                    job.job_id,
                    progress=job.progress,
                    message=job.message,
                    current_stage=job.current_stage,
                    pipeline_stage=job.pipeline_stage,
                    processed_frames=job.processed_frames,
                    total_frames=job.total_frames,
                    current_item_index=job.current_item_index,
                    total_items=job.total_items,
                    current_frame_timestamp=job.current_frame_timestamp,
                    total_duration=job.total_duration,
                    last_heartbeat=now,  # Update heartbeat on every persist
                    checkpoint_data=job.checkpoint_data,
                )
                self._last_db_update[job.job_id] = now
            except Exception:
                pass

    def update_pipeline_stage(
        self,
        job_id: str,
        stage: PipelineStage | str,
        current_item: int = 0,
        total_items: int = 0,
        message: str = "",
    ) -> None:
        """Update the pipeline stage for granular progress tracking.

        Args:
            job_id: Job ID to update.
            stage: Current pipeline stage.
            current_item: Current item index in this stage.
            total_items: Total items to process in this stage.
            message: Optional status message.
        """
        stage_value = stage.value if isinstance(stage, PipelineStage) else stage

        with self._lock:
            if job_id not in self._cache:
                return
            job = self._cache[job_id]
            job.pipeline_stage = stage_value
            job.current_item_index = current_item
            job.total_items = total_items
            if message:
                job.message = message
            job.last_heartbeat = time.time()

        # Broadcast stage update
        self._broadcast(
            {
                "event": "pipeline_stage_update",
                "job_id": job_id,
                "pipeline_stage": stage_value,
                "current_item": current_item,
                "total_items": total_items,
                "message": message,
            }
        )

        # Always persist stage changes immediately (important for crash recovery)
        try:
            job_manager.update_job(
                job_id,
                pipeline_stage=stage_value,
                current_item_index=current_item,
                total_items=total_items,
                last_heartbeat=time.time(),
                message=message if message else None,
            )
        except Exception:
            pass

    def save_checkpoint(self, job_id: str, data: dict[str, Any]) -> None:
        """Explicitly save key-value pairs to the job's checkpoint data.

        Useful for flags like 'audio_complete', 'voice_complete'.
        Does a safe merge with existing checkpoint data.
        """
        with self._lock:
            if job_id not in self._cache:
                return
            job = self._cache[job_id]

            # Merge with existing
            current = job.checkpoint_data or {}
            updated = {**current, **data}
            job.checkpoint_data = updated

            # Persist immediately
            try:
                job_manager.update_job(
                    job_id, checkpoint_data=updated, last_heartbeat=time.time()
                )
            except Exception:
                pass

        # Optional: Broadcast if needed (usually internal state)

    def complete(
        self, job_id: str, message: str = "Processing complete"
    ) -> None:
        """Mark job as complete."""
        with self._lock:
            if job_id in self._cache:
                job = self._cache[job_id]
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.completed_at = time.time()
                job.message = message

                # Immediate DB update
                job_manager.update_job(
                    job_id,
                    status=JobStatus.COMPLETED,
                    progress=100.0,
                    completed_at=job.completed_at,
                    message=message,
                )

        self._broadcast(
            {
                "event": "job_completed",
                "job_id": job_id,
                "message": message,
            }
        )

    def fail(self, job_id: str, error: str = "Unknown error") -> None:
        """Mark job as failed."""
        with self._lock:
            if job_id in self._cache:
                job = self._cache[job_id]
                job.status = JobStatus.FAILED
                job.progress = -1.0
                job.completed_at = time.time()
                job.error = error

                # Immediate DB update
                job_manager.update_job(
                    job_id,
                    status=JobStatus.FAILED,
                    progress=-1.0,
                    completed_at=job.completed_at,
                    error=error,
                )

        self._broadcast(
            {
                "event": "job_failed",
                "job_id": job_id,
                "error": error,
            }
        )

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        with self._lock:
            if job_id in self._cache:
                job = self._cache[job_id]
                if job.status in (JobStatus.RUNNING, JobStatus.PAUSED):
                    job.status = JobStatus.CANCELLED
                    job.completed_at = time.time()

                    # Immediate DB update
                    job_manager.update_job(
                        job_id,
                        status=JobStatus.CANCELLED,
                        completed_at=job.completed_at,
                    )

                    self._broadcast(
                        {
                            "event": "job_cancelled",
                            "job_id": job_id,
                        }
                    )
                    return True
        return False

    def pause(self, job_id: str) -> bool:
        """Pause a running job."""
        with self._lock:
            if job_id in self._cache:
                job = self._cache[job_id]
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.PAUSED

                    # Immediate DB update
                    job_manager.update_job(job_id, status=JobStatus.PAUSED)

                    self._broadcast({"event": "job_paused", "job_id": job_id})
                    return True
        return False

    def resume(self, job_id: str) -> bool:
        """Resume a paused job."""
        # Note: server.py will ensure the pipeline is restarted if needed
        # This method mainly updates status and broadcast
        with self._lock:
            if job_id in self._cache:
                job = self._cache[job_id]
                if job.status == JobStatus.PAUSED:
                    job.status = JobStatus.RUNNING
                    job_manager.update_job(job_id, status=JobStatus.RUNNING)
                    self._broadcast({"event": "job_resumed", "job_id": job_id})
                    return True

        # Check in DB if not in cache (e.g. after restart)
        job = job_manager.get_job(job_id)
        if job and job.status == JobStatus.PAUSED:
            with self._lock:
                self._cache[job_id] = job
                job.status = JobStatus.RUNNING

            job_manager.update_job(job_id, status=JobStatus.RUNNING)
            self._broadcast({"event": "job_resumed", "job_id": job_id})
            return True

        return False

    def delete(self, job_id: str) -> bool:
        """Delete a job from both cache and database.

        Args:
            job_id: The job ID to delete.

        Returns:
            True if the job was deleted, False if not found.
        """
        found = False

        with self._lock:
            if job_id in self._cache:
                del self._cache[job_id]
                found = True
            if job_id in self._last_db_update:
                del self._last_db_update[job_id]

        try:
            job_manager.delete_job(job_id)
            found = True
        except Exception:
            pass

        if found:
            self._broadcast({"event": "job_deleted", "job_id": job_id})

        return found

    def is_paused(self, job_id: str) -> bool:
        """Check if a job is paused."""
        with self._lock:
            job = self._cache.get(job_id)
            if job:
                return job.status == JobStatus.PAUSED
        # Fallback to DB check? No, cache should be sync'd.
        return False

    def is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled."""
        with self._lock:
            job = self._cache.get(job_id)
            if job:
                return job.status == JobStatus.CANCELLED
        return False

    def get(self, job_id: str) -> JobInfo | None:
        """Get job info by ID."""
        with self._lock:
            if job_id in self._cache:
                return self._cache[job_id]
        return job_manager.get_job(job_id)

    def get_all(self) -> list[JobInfo]:
        """Get all jobs."""
        # Refresh cache from DB to ensure we have latest history
        self._sync_cache()
        with self._lock:
            return sorted(
                self._cache.values(),
                key=lambda x: x.started_at,
                reverse=True,
            )

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to job events."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Unsubscribe from job events."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def emit_event(
        self,
        job_id: str,
        status: JobStatus | None = None,
        progress: float | None = None,
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit a custom event to SSE subscribers.

        Used for special events like job deletion that need to notify UI.
        """
        event: dict[str, Any] = {
            "event": "custom",
            "job_id": job_id,
            "message": message,
        }
        if status is not None:
            event["status"] = (
                status.value if isinstance(status, JobStatus) else status
            )
        if progress is not None:
            event["progress"] = progress
        if payload:
            event.update(payload)
        self._broadcast(event)

    def broadcast(self, event: dict[str, Any]) -> None:
        """Broadcast arbitrary event to SSE subscribers."""
        self._broadcast(event)

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
