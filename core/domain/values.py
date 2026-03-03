"""Domain value objects to eliminate Primitive Obsession code smells.

These strong types wrap raw `str`, `int`, and `float` values to guarantee
domain rules (e.g., timestamps can't be negative, job IDs have specific
formats) and drastically improve typing clarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class VideoPath:
    """Strong type for a video file path."""

    value: Path

    def __post_init__(self) -> None:
        """Validate path isn't empty."""
        if not str(self.value).strip():
            raise ValueError("Video path cannot be empty")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Timestamp:
    """Strong type representing points in time with guarantees."""

    seconds: float

    def __post_init__(self) -> None:
        """Validate time doesn't break physics."""
        if self.seconds < 0:
            raise ValueError(f"Timestamp cannot be negative: {self.seconds}")

    def __add__(self, other: Timestamp | float | int) -> Timestamp:
        val = other.seconds if isinstance(other, Timestamp) else other
        return Timestamp(self.seconds + val)

    def __sub__(self, other: Timestamp | float | int) -> Timestamp:
        val = other.seconds if isinstance(other, Timestamp) else other
        if self.seconds - val < 0:
            # Floor at zero rather than invalid state
            return Timestamp(0.0)
        return Timestamp(self.seconds - val)

    def __lt__(self, other: Timestamp) -> bool:
        return self.seconds < other.seconds

    def __le__(self, other: Timestamp) -> bool:
        return self.seconds <= other.seconds


@dataclass(frozen=True)
class JobId:
    """Strong type representing async job UUIDs."""

    value: str
    _uuid_pattern: ClassVar[re.Pattern] = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
    )

    def __post_init__(self) -> None:
        if not self._uuid_pattern.match(self.value) and not self.value.startswith("sync-"):
            # We allow sync- prefixed IDs for synchronous ad-hoc tests
            raise ValueError(f"Invalid Job ID format: {self.value}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ClusterId:
    """Strong type for Qdrant face/voice cluster numeric IDs."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"Cluster ID cannot be negative: {self.value}")

    def __int__(self) -> int:
        return self.value
