"""Domain value objects for the AI-Media-Indexer.

These value objects wrap primitive types to provide type safety,
validation, and explicit domain semantics, curing primitive obsession.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoPath:
    """Represents a validated path to a video file."""

    value: str

    def __post_init__(self):
        if not self.value:
            raise ValueError("VideoPath cannot be empty")
        # In a real system, we might verify path.exists() here,
        # but for performance/tests we just ensure it's a valid string.

    def as_path(self) -> Path:
        return Path(self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Timestamp:
    """Represents a specific point in time in a media file (seconds)."""

    seconds: float

    def __post_init__(self):
        if self.seconds < 0:
            raise ValueError("Timestamp cannot be negative")

    def __str__(self) -> str:
        return f"{self.seconds:.3f}s"

    def __float__(self) -> float:
        return self.seconds


@dataclass(frozen=True)
class ClusterId:
    """Represents a unique identifier for a biometric cluster (face/voice)."""

    value: int

    def __post_init__(self):
        if self.value < -1:  # -1 is typically used for 'unclustered' / noise
            raise ValueError("ClusterId cannot be less than -1")

    def __str__(self) -> str:
        return str(self.value)

    def __int__(self) -> int:
        return self.value


@dataclass(frozen=True)
class JobId:
    """Represents a unique identifier for an ingestion job."""

    value: str

    def __post_init__(self):
        if not self.value:
            raise ValueError("JobId cannot be empty")

    def __str__(self) -> str:
        return self.value
