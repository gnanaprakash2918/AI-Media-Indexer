"""Global progress tracking for ingestion jobs."""

from __future__ import annotations

from threading import Lock


class ProgressTracker:
    """Thread-safe progress tracking for jobs."""

    def __init__(self):
        """Initialize the progress tracker."""
        self._lock = Lock()
        self._progress: dict[str, float] = {}

    def start(self, job_id: str):
        """Start tracking a job (0%)."""
        with self._lock:
            self._progress[job_id] = 0.0

    def update(self, job_id: str, percent: float):
        """Update job progress (0-100)."""
        with self._lock:
            self._progress[job_id] = min(100.0, max(0.0, percent))

    def complete(self, job_id: str):
        """Mark job as complete (100%)."""
        with self._lock:
            self._progress[job_id] = 100.0

    def fail(self, job_id: str):
        """Mark job as failed (-1.0)."""
        with self._lock:
            self._progress[job_id] = -1.0

    def get(self, job_id: str) -> float | None:
        """Get the current progress of a job."""
        with self._lock:
            return self._progress.get(job_id)


progress_tracker = ProgressTracker()
