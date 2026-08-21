"""Global progress tracking for ingestion jobs using PostgreSQL ChunkStateRepository."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

from core.storage.repositories.chunk_state_repo import ChunkStateRepository
from config import settings
from core.utils.logger import logger

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STUCK = "stuck"

@dataclass
class JobInfo:
    job_id: str
    file_path: str = ""
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    media_type: str = "unknown"
    started_at: float = 0.0
    completed_at: float | None = None
    error: str | None = None
    current_stage: str = ""
    pipeline_stage: str = "init"
    processed_frames: int = 0
    total_frames: int = 0
    current_item_index: int = 0
    total_items: int = 0
    current_frame_timestamp: float = 0.0
    total_duration: float = 0.0
    last_heartbeat: float = 0.0
    checkpoint_data: dict[str, Any] | None = None
    stage_stats: dict[str, Any] = field(default_factory=dict)

class ProgressTracker:
    def __init__(self):
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._paused: set[str] = set()
        self._cancelled: set[str] = set()
        self._live_jobs: dict[str, JobInfo] = {}

    def _broadcast_sync(self, data: dict[str, Any]):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast(data))
        except RuntimeError:
            pass # No running event loop

    def start(self, job_id: str, file_path: str = "", media_type: str = "video"):
        job = JobInfo(job_id=job_id, file_path=file_path, media_type=media_type, status=JobStatus.RUNNING)
        self._live_jobs[job_id] = job
        self._broadcast_sync({"type": "job_started", "job": job.__dict__})

    def update(self, job_id: str, progress: float, **kwargs):
        job = self._live_jobs.get(job_id)
        if job:
            job.progress = progress
            for k, v in kwargs.items():
                setattr(job, k, v)
            self._broadcast_sync({"type": "job_progress", "job": job.__dict__})
        
    def fail(self, job_id: str, error: str):
        job = self._live_jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error = error
            self._broadcast_sync({"type": "job_failed", "job": job.__dict__})
        
    def complete(self, job_id: str, message: str = ""):
        job = self._live_jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.message = message
            job.progress = 100.0
            self._broadcast_sync({"type": "job_completed", "job": job.__dict__})

    @asynccontextmanager
    async def stage(self, job_id: str, stage_name: str, **kwargs) -> AsyncGenerator[None, None]:
        self.stage_start(job_id, stage_name)
        try:
            yield
            self.stage_complete(job_id, stage_name)
        except Exception as e:
            self.fail(job_id, str(e))
            raise

    def stage_start(self, job_id: str, stage_name: str):
        job = self._live_jobs.get(job_id)
        if job:
            job.current_stage = stage_name
            self._broadcast_sync({"type": "stage_start", "job_id": job_id, "stage": stage_name})

    def stage_complete(self, job_id: str, stage_name: str, message: str = ""):
        job = self._live_jobs.get(job_id)
        if job:
            self._broadcast_sync({"type": "stage_complete", "job_id": job_id, "stage": stage_name, "message": message})
        
    def save_checkpoint(self, job_id: str, data: dict[str, Any]):
        self._checkpoints[job_id] = data
        job = self._live_jobs.get(job_id)
        if job:
            job.checkpoint_data = data

    def get_checkpoint(self, job_id: str) -> dict[str, Any] | None:
        return self._checkpoints.get(job_id)

    def increment_retry(self, job_id: str, stage: str, error: str):
        pass

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled

    def is_paused(self, job_id: str) -> bool:
        return job_id in self._paused
        
    def cancel(self, job_id: str) -> bool:
        self._cancelled.add(job_id)
        return True

    def pause(self, job_id: str) -> bool:
        self._paused.add(job_id)
        return True

    def resume(self, job_id: str) -> bool:
        self._paused.discard(job_id)
        return True

    def delete(self, job_id: str) -> bool:
        self._checkpoints.pop(job_id, None)
        self._live_jobs.pop(job_id, None)
        self._paused.discard(job_id)
        self._cancelled.discard(job_id)
        return True

    async def get(self, job_id: str) -> JobInfo | None:
        # Check live jobs cache first for real-time UI
        if job_id in self._live_jobs:
            return self._live_jobs[job_id]
            
        stats = await self.get_job_stats(job_id)
        if not stats or stats.get("total_chunks", 0) == 0:
            return None
        return JobInfo(
            job_id=job_id, 
            status=JobStatus(stats["status"]), 
            progress=stats["progress"],
            checkpoint_data=self._checkpoints.get(job_id)
        )

    async def get_all(self) -> list[JobInfo]:
        try:
            async with ChunkStateRepository.from_settings() as repo:
                media_ids = await repo.get_all_active_media_ids()
                
            jobs = []
            for m_id in media_ids:
                job = await self.get(m_id)
                if job:
                    jobs.append(job)
            return jobs
        except Exception as e:
            logger.warning(f"Error fetching all jobs: {e}")
            return []

    async def get_job_stats(self, job_id: str) -> dict[str, Any]:
        try:
            async with ChunkStateRepository.from_settings() as repo:
                return await repo.get_progress_for_media(job_id)
        except Exception as e:
            logger.warning(f"Error fetching stats for job {job_id}: {e}")
            return {"status": "failed", "progress": 0.0}

    async def broadcast(self, data: dict[str, Any]):
        for q in self._subscribers:
            await q.put(data)

    async def listen(self):
        q = asyncio.Queue()
        self._subscribers.append(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._subscribers.remove(q)

progress_tracker = ProgressTracker()
