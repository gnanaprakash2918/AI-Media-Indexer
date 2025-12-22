"""Utilities for sampling frames from video."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import settings


@dataclass(slots=True)
class FrameSampler:
    """Helper to determine whether a frame should be sampled."""

    every_n: int = field(default_factory=lambda: settings.frame_sample_ratio)

    def should_sample(self, index: int) -> bool:
        """Check if the frame at `index` should be processed."""
        return index % self.every_n == 0
