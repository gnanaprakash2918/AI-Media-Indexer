"""Global progress tracking for ingestion jobs with SSE support and persistence."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from threading import Lock
from typing import Any

from core.ingestion.jobs import JobInfo, JobStatus, PipelineStage, job_manager

STAGE_WEIGHTS = {
    "init": 0.01,
    "audio": 0.15,
    "voice": 0.12,
    "audio_events": 0.05,
    "frames": 0.35,
    "frames_chunk_0": 0.35,
    "scene_captions": 0.15,
    "scene_captions_chunk_0": 0.15,
    "post_processing": 0.05,
    "index": 0.10,
    "complete": 0.02,
}

ORDERED_STAGES = [
    "init", "audio", "voice", "audio_events",
    "frames", "scene_captions", "post_processing", "complete"
]


class ProgressTracker:
    """Thread-safe progress tracking with SSE broadcasting and SQLite persistence."""

    def __init__(self) -> None:
        """Initialize the progress tracker."""
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._cache: dict[str, JobInfo] = {}
        self._last_db_update: dict[str, float] = {}
        self._lock = Lock()

        self._event_id_counter = 0
        self._recent_events: list[dict[str, Any]] = []
        self._max_recent_events = 100

        self._speed_data: dict[str, dict[str, Any]] = {}

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

    def _calculate_weighted_progress(self, job: JobInfo) -> float:
        """Calculate weighted progress based on completed stages."""
        total = 0.0
        stage_stats = job.stage_stats or {}

        for stage, weight in STAGE_WEIGHTS.items():
            stats = stage_stats.get(stage, {})
            status = stats.get("status", "pending")

            if status == "completed":
                total += weight * 100
            elif status == "running":
                total += weight * 50
            elif status == "skipped":
                total += weight * 100

        return min(100.0, total)

    def _update_speed(
        self,
        job_id: str,
        frames_processed: int | None = None,
        timestamp_processed: float | None = None,
    ) -> dict[str, Any]:
        """Track and calculate processing speed metrics."""
        now = time.time()

        if job_id not in self._speed_data:
            self._speed_data[job_id] = {
                "start_time": now,
                "last_update": now,
                "last_frames": 0,
                "last_timestamp": 0.0,
                "fps": 0.0,
                "speed_ratio": 0.0,
            }

        data = self._speed_data[job_id]
        elapsed_since_last = now - data["last_update"]

        if elapsed_since_last >= 2.0:
            if frames_processed is not None:
                frame_delta = frames_processed - data["last_frames"]
                if frame_delta > 0 and elapsed_since_last > 0:
                    data["fps"] = frame_delta / elapsed_since_last
                data["last_frames"] = frames_processed

            if timestamp_processed is not None:
                ts_delta = timestamp_processed - data["last_timestamp"]
                if ts_delta > 0 and elapsed_since_last > 0:
                    data["speed_ratio"] = ts_delta / elapsed_since_last
                data["last_timestamp"] = timestamp_processed

            data["last_update"] = now

        return {
            "fps": round(data["fps"], 1),
            "speed_ratio": round(data["speed_ratio"], 2),
        }

    def _calculate_eta(self, job: JobInfo) -> float | None:
        """Calculate estimated time remaining in seconds."""
        if job.total_duration <= 0 or job.current_frame_timestamp <= 0:
            return None

        remaining_video = job.total_duration - job.current_frame_timestamp

        speed_data = self._speed_data.get(job.job_id, {})
        speed_ratio = speed_data.get("speed_ratio", 0)

        if speed_ratio > 0:
            return remaining_video / speed_ratio

        elapsed = time.time() - job.started_at
        if elapsed > 10 and job.current_frame_timestamp > 0:
            estimated_ratio = job.current_frame_timestamp / elapsed
            if estimated_ratio > 0:
                return remaining_video / estimated_ratio

        return None

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

            # Simple 0-100 updates
            if percent >= 0:
                job.progress = min(100.0, max(0.0, percent))

            if stage:
                job.current_stage = stage
                # Also update pipeline_stage so frontend displays correct stage
                job.pipeline_stage = stage
            if message:
                job.message = message

            # Broadcast immediately
            self._broadcast(
                {
                    "event": "job_progress",
                    "job_id": job_id,
                    "progress": job.progress,
                    "stage": job.current_stage,
                    "pipeline_stage": job.pipeline_stage,
                    "message": job.message,
                }
            )

            # Persist to DB (throttled)
            self._persist_throttled(job)

    def stage_start(
        self, job_id: str, stage_name: str, message: str = ""
    ) -> None:
        """Mark the start of a processing stage."""
        with self._lock:
            if job_id not in self._cache:
                return
            job = self._cache[job_id]

            job.current_stage = stage_name
            job.pipeline_stage = stage_name
            if message:
                job.message = message

            # Initialize or update stage stats
            current_stats = job.stage_stats.get(stage_name, {})
            current_stats.update(
                {
                    "status": "running",
                    "start": time.time(),
                    "end": None,
                    "duration": None,
                }
            )
            job.stage_stats[stage_name] = current_stats

            # Broadcast start
            self._broadcast(
                {
                    "event": "stage_start",
                    "job_id": job_id,
                    "stage": stage_name,
                    "message": message,
                    "timestamp": time.time(),
                }
            )

            # Persist stage update immediately
            self._persist_throttled(job, force=True)

    def stage_complete(
        self, job_id: str, stage_name: str, message: str = "Complete"
    ) -> None:
        """Mark a stage as completed."""
        with self._lock:
            if job_id not in self._cache:
                return
            job = self._cache[job_id]

            end_time = time.time()

            if stage_name in job.stage_stats:
                stats = job.stage_stats[stage_name]
                start_time = stats.get("start", end_time)
                duration = end_time - start_time

                stats.update(
                    {
                        "status": "completed",
                        "end": end_time,
                        "duration": duration,
                    }
                )
                job.stage_stats[stage_name] = stats

            if message:
                job.message = message

            # Broadcast completion
            self._broadcast(
                {
                    "event": "stage_complete",
                    "job_id": job_id,
                    "stage": stage_name,
                    "duration": job.stage_stats.get(stage_name, {}).get(
                        "duration"
                    ),
                    "message": message,
                }
            )

            # Persist immediately
            self._persist_throttled(job, force=True)

    def stage_fail(self, job_id: str, stage_name: str, error: str) -> None:
        """Mark a stage as failed."""
        with self._lock:
            if job_id not in self._cache:
                return
            job = self._cache[job_id]

            if stage_name in job.stage_stats:
                job.stage_stats[stage_name].update(
                    {"status": "failed", "end": time.time(), "error": error}
                )

            # Also fail the job overall
            self.fail(job_id, error=error)

    def increment_retry(self, job_id: str, stage_name: str) -> None:
        """Increment the retry count for a stage."""
        with self._lock:
            if job_id not in self._cache:
                return
            job = self._cache[job_id]

            if stage_name in job.stage_stats:
                stats = job.stage_stats[stage_name]
                stats["retries"] = stats.get("retries", 0) + 1
                job.stage_stats[stage_name] = stats

            # Persist (throttled)
            self._persist_throttled(job)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def stage(self, job_id: str, stage_name: str, message: str = ""):
        """Context manager for tracking a pipeline stage."""
        self.stage_start(job_id, stage_name, message)
        try:
            yield
            self.stage_complete(
                job_id, stage_name, message=f"{stage_name} complete"
            )
        except Exception as e:
            self.stage_fail(job_id, stage_name, error=str(e))
            raise

    def update_granular(
        self,
        job_id: str,
        processed_frames: int | None = None,
        total_frames: int | None = None,
        current_timestamp: float | None = None,
        total_duration: float | None = None,
    ) -> None:
        """Update granular progress details and track speed."""
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

            # Update speed tracking
            speed = self._update_speed(
                job_id,
                frames_processed=processed_frames,
                timestamp_processed=current_timestamp,
            )

            # Create checkpoint data for resuming (merge with existing)
            if current_timestamp is not None:
                existing_checkpoint = job.checkpoint_data or {}
                new_checkpoint = {
                    **existing_checkpoint,
                    "timestamp": current_timestamp,
                    "processed_frames": job.processed_frames,
                }
                job.checkpoint_data = new_checkpoint

            # Calculate ETA for broadcast
            eta = self._calculate_eta(job)

            # Broadcast detailed update with speed and ETA
            self._broadcast(
                {
                    "event": "job_granular_update",
                    "job_id": job_id,
                    "processed_frames": job.processed_frames,
                    "total_frames": job.total_frames,
                    "timestamp": job.current_frame_timestamp,
                    "duration": job.total_duration,
                    "speed": speed,
                    "eta_seconds": eta,
                }
            )

            # Persist to DB (throttled)
            self._persist_throttled(job)

    def _persist_throttled(self, job: JobInfo, force: bool = False) -> None:
        """Persist job state to DB if enough time has passed."""
        import time as time_module

        now = time_module.time()
        last = self._last_db_update.get(job.job_id, 0)

        # Update DB every 2 seconds max for progress to reduce I/O, unless forced
        if force or (now - last > 2.0):
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
                    stage_stats=job.stage_stats,
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
                job.pipeline_stage = "complete"
                job.current_stage = "complete"

                # Immediate DB update
                job_manager.update_job(
                    job_id,
                    status=JobStatus.COMPLETED,
                    progress=100.0,
                    completed_at=job.completed_at,
                    message=message,
                    pipeline_stage="complete",
                    current_stage="complete",
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
        """Broadcast event to all subscribers with event ID for reconnection."""
        self._event_id_counter += 1
        event["event_id"] = self._event_id_counter
        event["timestamp"] = time.time()

        self._recent_events.append(event)
        if len(self._recent_events) > self._max_recent_events:
            self._recent_events = self._recent_events[-self._max_recent_events:]

        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def get_events_since(self, last_event_id: int) -> list[dict[str, Any]]:
        """Get events since a given event ID for reconnection replay."""
        return [e for e in self._recent_events if e.get("event_id", 0) > last_event_id]

    async def event_stream(
        self, last_event_id: int | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator for SSE events with heartbeat.

        Args:
            last_event_id: If provided, replay events since this ID.
        """
        queue = self.subscribe()

        if last_event_id is not None:
            missed = self.get_events_since(last_event_id)
            for event in missed:
                yield event

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield event
                except asyncio.TimeoutError:
                    yield {
                        "event": "heartbeat",
                        "event_id": self._event_id_counter,
                        "timestamp": time.time(),
                    }
        finally:
            self.unsubscribe(queue)

    def get_job_stats(self, job_id: str) -> dict[str, Any] | None:
        """Get full job stats including computed fields for API response."""
        job = self.get(job_id)
        if not job:
            return None

        weighted_progress = self._calculate_weighted_progress(job)
        eta = self._calculate_eta(job)
        speed = self._speed_data.get(job_id, {})

        return {
            "job_id": job.job_id,
            "status": job.status.value if hasattr(job.status, "value") else job.status,
            "progress": job.progress,
            "weighted_progress": round(weighted_progress, 1),
            "file_path": job.file_path,
            "media_type": job.media_type,
            "current_stage": job.current_stage,
            "pipeline_stage": getattr(job, "pipeline_stage", "init"),
            "message": job.message,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error": job.error,
            "total_frames": job.total_frames,
            "processed_frames": job.processed_frames,
            "current_item_index": getattr(job, "current_item_index", 0),
            "total_items": getattr(job, "total_items", 0),
            "timestamp": job.current_frame_timestamp,
            "duration": job.total_duration,
            "last_heartbeat": getattr(job, "last_heartbeat", 0.0),
            "stage_stats": job.stage_stats or {},
            "eta_seconds": eta,
            "speed": {
                "fps": speed.get("fps", 0),
                "speed_ratio": speed.get("speed_ratio", 0),
            },
            "checkpoint_data": job.checkpoint_data,
        }


progress_tracker = ProgressTracker()
