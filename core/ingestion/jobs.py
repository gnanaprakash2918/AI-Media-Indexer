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
    STUCK = "stuck"        # Detected as stuck (heartbeat timeout)


class PipelineStage(str, Enum):
    """Granular pipeline stages for accurate progress tracking."""
    INIT = "init"              # Initial setup
    EXTRACT = "extract"        # Extracting audio/frames
    TRANSCRIBE = "transcribe"  # Whisper transcription
    DIARIZE = "diarize"        # Speaker diarization
    FACE_DETECT = "face_detect"  # Face detection
    FACE_TRACK = "face_track"    # Track-level face clustering
    VOICE_EMBED = "voice_embed"  # Voice embedding extraction
    VLM_CAPTION = "vlm_caption"  # VLM dense captioning
    INDEX = "index"            # Vector DB indexing
    COMPLETE = "complete"      # All stages done

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
    pipeline_stage: str = "init"  # PipelineStage enum value
    
    # Granular Progress Details
    processed_frames: int = 0
    total_frames: int = 0
    current_item_index: int = 0   # Generic item counter (frames, segments, etc)
    total_items: int = 0          # Total items in current stage
    current_frame_timestamp: float = 0.0
    total_duration: float = 0.0
    
    # Crash recovery
    last_heartbeat: float = 0.0   # Last heartbeat timestamp
    
    # Checkpoint data for resuming
    checkpoint_data: dict[str, Any] | None = None

class JobManager:
    """SQLite-backed job manager for persistent state."""

    def __init__(self, db_path: str = "jobs.db") -> None:
        self.db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database with migration support."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for crash safety
            conn.execute("PRAGMA journal_mode = WAL")
            
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
                    pipeline_stage TEXT DEFAULT 'init',
                    processed_frames INTEGER DEFAULT 0,
                    total_frames INTEGER DEFAULT 0,
                    current_item_index INTEGER DEFAULT 0,
                    total_items INTEGER DEFAULT 0,
                    current_frame_timestamp REAL DEFAULT 0.0,
                    total_duration REAL DEFAULT 0.0,
                    last_heartbeat REAL DEFAULT 0.0,
                    checkpoint_data TEXT,
                    created_at REAL DEFAULT (unixepoch())
                )
            """)
            
            # Migration: Add new columns if they don't exist
            self._migrate_schema(conn)
            conn.commit()
    
    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Add new columns to existing tables (safe migration)."""
        cursor = conn.execute("PRAGMA table_info(jobs)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        migrations = [
            ("pipeline_stage", "TEXT DEFAULT 'init'"),
            ("current_item_index", "INTEGER DEFAULT 0"),
            ("total_items", "INTEGER DEFAULT 0"),
            ("last_heartbeat", "REAL DEFAULT 0.0"),
        ]
        
        for col_name, col_def in migrations:
            if col_name not in existing_columns:
                try:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_def}")
                    logger.info(f"Migrated jobs table: added {col_name}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

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
            "current_stage", "pipeline_stage", "processed_frames", "total_frames",
            "current_item_index", "total_items", "current_frame_timestamp", 
            "total_duration", "last_heartbeat", "checkpoint_data"
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
            current_item_index=row["current_item_index"] if "current_item_index" in row.keys() else 0,
            total_items=row["total_items"] if "total_items" in row.keys() else 0,
            current_frame_timestamp=row["current_frame_timestamp"],
            total_duration=row["total_duration"],
            pipeline_stage=row["pipeline_stage"] if "pipeline_stage" in row.keys() else "init",
            last_heartbeat=row["last_heartbeat"] if "last_heartbeat" in row.keys() else 0.0,
            checkpoint_data=checkpoint
        )
    
    def update_heartbeat(self, job_id: str) -> None:
        """Update job heartbeat to current time (call periodically during processing)."""
        now = time.time()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET last_heartbeat = ? WHERE job_id = ?",
                (now, job_id)
            )
            conn.commit()
    
    def detect_stuck_jobs(self, timeout_seconds: float = 60.0) -> list[JobInfo]:
        """Detect jobs that are RUNNING but haven't sent a heartbeat in timeout_seconds.
        
        Args:
            timeout_seconds: Heartbeat timeout (default 60s = 1 minute).
            
        Returns:
            List of stuck jobs.
        """
        cutoff = time.time() - timeout_seconds
        stuck_jobs = []
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM jobs 
                WHERE status = 'running' 
                AND last_heartbeat > 0 
                AND last_heartbeat < ?
            """, (cutoff,))
            
            stuck_jobs = [self._row_to_job(row) for row in cursor.fetchall()]
        
        return stuck_jobs
    
    def mark_stuck_jobs(self, timeout_seconds: float = 60.0) -> int:
        """Mark stuck jobs as STUCK status.
        
        Args:
            timeout_seconds: Heartbeat timeout.
            
        Returns:
            Number of jobs marked as stuck.
        """
        stuck = self.detect_stuck_jobs(timeout_seconds)
        for job in stuck:
            self.update_job(job.job_id, status=JobStatus.STUCK)
            logger.warning(f"Marked job {job.job_id} as STUCK (no heartbeat for {timeout_seconds}s)")
        return len(stuck)
    
    def recover_on_startup(self, timeout_seconds: float = 60.0) -> dict[str, int]:
        """Auto-detect and handle stuck jobs on application startup.
        
        Called during server initialization to recover from crashes.
        
        Strategy:
        - Jobs with status=RUNNING but stale heartbeat → mark as PAUSED (resumable)
        - Jobs with status=STUCK → leave as is for manual review
        
        Args:
            timeout_seconds: Heartbeat timeout.
            
        Returns:
            Dict with recovery stats.
        """
        stats = {"paused": 0, "stuck": 0}
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cutoff = time.time() - timeout_seconds
            
            # Find running jobs with stale heartbeat
            cursor = conn.execute("""
                SELECT * FROM jobs 
                WHERE status = 'running' 
                AND (last_heartbeat < ? OR last_heartbeat = 0)
            """, (cutoff,))
            
            for row in cursor.fetchall():
                job_id = row["job_id"]
                # Auto-pause for resumption
                conn.execute(
                    "UPDATE jobs SET status = 'paused' WHERE job_id = ?",
                    (job_id,)
                )
                stats["paused"] += 1
                logger.info(f"Recovered job {job_id}: marked as PAUSED (was running without heartbeat)")
            
            conn.commit()
        
        return stats


# Global instance
job_manager = JobManager()
