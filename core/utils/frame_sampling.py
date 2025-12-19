"""Utilities for sampling frames from video."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FrameSampler:
    """Helper to determine whether a frame should be sampled."""

    every_n: int = 5

    def should_sample(self, index: int) -> bool:
        """Check if the frame at `index` should be processed."""
        return index % self.every_n == 0
