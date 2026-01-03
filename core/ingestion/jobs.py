"""Persistent job management using SQLite."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from config import settings
from core.utils.logger import logger

class JobStatus(str, Enum):
    """Status of a processing job."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"      # Persistent pause (process stopped, can resume)
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class JobInfo:
    """Information about a processing job."""
    job_id: str
    file_path: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    media_type: str = "unknown"
    started_at: float = 0.0
    completed_at: float | None = None
    error: str | None = None
    current_stage: str = ""
    
    # Granular Progress Details
    processed_frames: int = 0
    total_frames: int = 0
    current_frame_timestamp: float = 0.0
    total_duration: float = 0.0
    
    # Checkpoint data for resuming
    checkpoint_data: dict[str, Any] | None = None

class JobManager:
    """SQLite-backed job manager for persistent state."""

    def __init__(self, db_path: str = "jobs.db") -> None:
        self.db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    message TEXT DEFAULT '',
                    media_type TEXT DEFAULT 'unknown',
                    started_at REAL DEFAULT 0.0,
                    completed_at REAL,
                    error TEXT,
                    current_stage TEXT DEFAULT '',
                    processed_frames INTEGER DEFAULT 0,
                    total_frames INTEGER DEFAULT 0,
                    current_frame_timestamp REAL DEFAULT 0.0,
                    total_duration REAL DEFAULT 0.0,
                    checkpoint_data TEXT,
                    created_at REAL DEFAULT (unixepoch())
                )
            """)
            conn.commit()

    def create_job(self, job_id: str, file_path: str, media_type: str = "unknown") -> JobInfo:
        """Create a new job."""
        job = JobInfo(
            job_id=job_id,
            file_path=file_path,
            media_type=media_type,
            started_at=time.time(),
            status=JobStatus.RUNNING # Starts running immediately in our model
        )
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, file_path, status, media_type, started_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (job.job_id, job.file_path, job.status.value, job.media_type, job.started_at)
            )
            conn.commit()
        return job

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        """Update job fields."""
        valid_fields = {
            "status", "progress", "message", "completed_at", "error", 
            "current_stage", "processed_frames", "total_frames", 
            "current_frame_timestamp", "total_duration", "checkpoint_data"
        }
        
        updates = []
        values = []
        
        for k, v in kwargs.items():
            if k in valid_fields:
                if k == "status" and isinstance(v, JobStatus):
                    v = v.value
                elif k == "checkpoint_data" and isinstance(v, dict):
                    v = json.dumps(v)
                updates.append(f"{k} = ?")
                values.append(v)
        
        if not updates:
            return

        values.append(job_id)
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?",
                values
            )
            conn.commit()

    def get_job(self, job_id: str) -> JobInfo | None:
        """Get job by ID."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_job(row)

    def get_all_jobs(self, limit: int = 50) -> list[JobInfo]:
        """Get all jobs, most recent first."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._row_to_job(row) for row in cursor.fetchall()]

    def delete_job(self, job_id: str) -> None:
        """Delete a job."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()

    def _row_to_job(self, row: sqlite3.Row) -> JobInfo:
        """Convert DB row to JobInfo."""
        checkpoint = None
        if row["checkpoint_data"]:
            try:
                checkpoint = json.loads(row["checkpoint_data"])
            except Exception:
                pass

        return JobInfo(
            job_id=row["job_id"],
            file_path=row["file_path"],
            status=JobStatus(row["status"]),
            progress=row["progress"],
            message=row["message"],
            media_type=row["media_type"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error=row["error"],
            current_stage=row["current_stage"],
            processed_frames=row["processed_frames"],
            total_frames=row["total_frames"],
            current_frame_timestamp=row["current_frame_timestamp"],
            total_duration=row["total_duration"],
            checkpoint_data=checkpoint
        )

# Global instance
job_manager = JobManager()
