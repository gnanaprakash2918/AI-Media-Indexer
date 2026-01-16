"""Preemptive cancellation pattern for GPU-heavy jobs.

Enables "Stop Job" to actually stop within seconds, not minutes.
Uses cooperative checkpointing - call token.check() at safe points.
"""

from dataclasses import dataclass, field
from typing import Callable


class CancellationError(Exception):
    """Raised when a job is cancelled mid-execution."""

    pass


@dataclass
class CancellationToken:
    """Token for cooperative job cancellation.

    Usage:
        token = get_or_create_token(job_id)
        for i, frame in enumerate(frames):
            if i % 5 == 0:
                token.check()  # Raises CancellationError if cancelled
            process(frame)
    """

    job_id: str
    _cancelled: bool = False
    _callbacks: list[Callable[[], None]] = field(default_factory=list)

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True
        for callback in self._callbacks:
            try:
                callback()
            except Exception:
                pass  # Don't let callback errors block cancellation

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register a callback to run on cancellation.

        Args:
            callback: Function to call when job is cancelled.
        """
        self._callbacks.append(callback)

    def check(self) -> None:
        """Check if cancelled and raise if so.

        Call this at checkpoints in long-running operations.

        Raises:
            CancellationError: If job has been cancelled.
        """
        if self._cancelled:
            raise CancellationError(f"Job {self.job_id} was cancelled")


# Global registry of active cancellation tokens
_ACTIVE_TOKENS: dict[str, CancellationToken] = {}


def get_or_create_token(job_id: str) -> CancellationToken:
    """Get existing token or create new one for a job.

    Args:
        job_id: Unique job identifier.

    Returns:
        CancellationToken for the job.
    """
    if job_id not in _ACTIVE_TOKENS:
        _ACTIVE_TOKENS[job_id] = CancellationToken(job_id=job_id)
    return _ACTIVE_TOKENS[job_id]


def cancel_job(job_id: str) -> bool:
    """Cancel a job by its ID.

    Args:
        job_id: Job to cancel.

    Returns:
        True if token existed and was cancelled.
    """
    if job_id in _ACTIVE_TOKENS:
        _ACTIVE_TOKENS[job_id].cancel()
        return True
    return False


def cleanup_token(job_id: str) -> None:
    """Remove a token after job completion.

    Args:
        job_id: Job that completed.
    """
    _ACTIVE_TOKENS.pop(job_id, None)


def list_active_jobs() -> list[str]:
    """List all active job IDs.

    Returns:
        List of job IDs with active tokens.
    """
    return list(_ACTIVE_TOKENS.keys())


def is_job_cancelled(job_id: str) -> bool:
    """Check if a specific job is cancelled.

    Args:
        job_id: Job to check.

    Returns:
        True if job exists and is cancelled.
    """
    if job_id in _ACTIVE_TOKENS:
        return _ACTIVE_TOKENS[job_id].is_cancelled
    return False
