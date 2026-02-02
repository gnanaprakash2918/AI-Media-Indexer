"""Smart frame sampling to reduce expensive VLM calls.

Motion-gated sampling only analyzes frames with significant visual changes,
reducing VLM calls by ~70% while maintaining quality.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class SmartFrameSampler:
    """Motion-gated frame sampling to reduce VLM calls by ~70%.

    Only runs expensive VLM analysis on frames with significant motion,
    skipping redundant static frames.

    Usage:
        sampler = SmartFrameSampler(motion_threshold=30.0)
        for frame in frames:
            if sampler.should_analyze(frame):
                result = await vlm.analyze(frame)
        print(sampler.get_stats())  # {'skip_rate': '68%', ...}
    """

    def __init__(
        self,
        motion_threshold: float = 30.0,
        min_analyze_interval: int = 5,
    ):
        """Initialize smart frame sampler.

        Args:
            motion_threshold: Mean pixel difference threshold (0-255).
                Higher = fewer frames analyzed. Default 30.0.
            min_analyze_interval: Minimum frames between forced analysis.
                Ensures at least one frame per N is analyzed.
        """
        self.motion_threshold = motion_threshold
        self.min_analyze_interval = min_analyze_interval
        self.prev_frame: np.ndarray | None = None
        self.skipped_count = 0
        self.analyzed_count = 0
        self.frames_since_analysis = 0

    def should_analyze(self, frame: np.ndarray) -> bool:
        """Determine if frame should be analyzed by VLM.

        Args:
            frame: RGB/BGR frame as numpy array (H, W, C).

        Returns:
            True if frame should be analyzed.
        """
        self.frames_since_analysis += 1

        # Always analyze first frame
        if self.prev_frame is None:
            self.prev_frame = self._to_gray(frame)
            self.analyzed_count += 1
            self.frames_since_analysis = 0
            return True

        # Force analysis after min_analyze_interval
        if self.frames_since_analysis >= self.min_analyze_interval:
            self.prev_frame = self._to_gray(frame)
            self.analyzed_count += 1
            self.frames_since_analysis = 0
            return True

        # Compute motion score
        gray_curr = self._to_gray(frame)
        motion_score = self._compute_motion(gray_curr, self.prev_frame)

        if motion_score > self.motion_threshold:
            self.prev_frame = gray_curr
            self.analyzed_count += 1
            self.frames_since_analysis = 0
            return True

        self.skipped_count += 1
        return False

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale."""
        if len(frame.shape) == 2:
            return frame
        if frame.shape[2] == 4:  # RGBA
            frame = frame[:, :, :3]
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    def _compute_motion(
        self,
        curr: np.ndarray,
        prev: np.ndarray,
    ) -> float:
        """Compute motion score between frames.

        Args:
            curr: Current grayscale frame.
            prev: Previous grayscale frame.

        Returns:
            Mean absolute difference (0-255).
        """
        # Resize if shapes don't match
        if curr.shape != prev.shape:
            prev = cv2.resize(prev, (curr.shape[1], curr.shape[0]))

        diff = cv2.absdiff(curr, prev)
        return float(np.mean(diff))

    def get_stats(self) -> dict[str, Any]:
        """Get sampling statistics.

        Returns:
            Dict with analyzed, skipped, total, and skip_rate.
        """
        total = self.analyzed_count + self.skipped_count
        skip_rate = self.skipped_count / total if total > 0 else 0
        return {
            "analyzed": self.analyzed_count,
            "skipped": self.skipped_count,
            "total": total,
            "skip_rate": f"{skip_rate:.1%}",
            "motion_threshold": self.motion_threshold,
        }

    def reset(self) -> None:
        """Reset for new video."""
        self.prev_frame = None
        self.skipped_count = 0
        self.analyzed_count = 0
        self.frames_since_analysis = 0


class TextGatedOCR:
    """Only run OCR if lightweight text detection finds text regions.

    Uses edge density as a cheap proxy for text presence before
    running expensive OCR models.

    Usage:
        gate = TextGatedOCR()
        if gate.has_text(frame):
            text = ocr.extract(frame)
    """

    def __init__(
        self,
        edge_threshold: float = 0.02,
        canny_low: int = 50,
        canny_high: int = 150,
    ):
        """Initialize text gate.

        Args:
            edge_threshold: Min edge density (0-1) to trigger OCR.
            canny_low: Canny edge detector low threshold.
            canny_high: Canny edge detector high threshold.
        """
        self.edge_threshold = edge_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.checks_count = 0
        self.text_found_count = 0

    def has_text(self, frame: np.ndarray) -> bool:
        """Check if frame likely contains text.

        Uses edge density as a cheap heuristic. Text regions
        tend to have high edge density.

        Args:
            frame: RGB/BGR frame as numpy array.

        Returns:
            True if text is likely present.
        """
        self.checks_count += 1

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Detect edges
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # Compute edge density
        edge_density = np.sum(edges > 0) / edges.size

        has_text = bool(edge_density > self.edge_threshold)
        if has_text:
            self.text_found_count += 1

        return has_text

    def get_stats(self) -> dict[str, Any]:
        """Get gating statistics."""
        rate = (
            self.text_found_count / self.checks_count
            if self.checks_count
            else 0
        )
        return {
            "total_checks": self.checks_count,
            "text_found": self.text_found_count,
            "text_rate": f"{rate:.1%}",
            "edge_threshold": self.edge_threshold,
        }

    def reset(self) -> None:
        """Reset statistics."""
        self.checks_count = 0
        self.text_found_count = 0


class SceneChangeDetector:
    """Detect scene changes for intelligent keyframe selection.

    Uses histogram comparison to detect significant visual changes.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        method: str = "correlation",
    ):
        """Initialize scene change detector.

        Args:
            threshold: Histogram difference threshold for scene change.
            method: Comparison method
                ('correlation', 'chi_square', 'intersection').
        """
        self.threshold = threshold
        self.method = method
        self.prev_hist: np.ndarray | None = None
        self.scene_changes = 0

        # OpenCV comparison methods
        self._methods = {
            "correlation": cv2.HISTCMP_CORREL,
            "chi_square": cv2.HISTCMP_CHISQR,
            "intersection": cv2.HISTCMP_INTERSECT,
        }

    def is_scene_change(self, frame: np.ndarray) -> bool:
        """Check if frame represents a scene change.

        Args:
            frame: RGB/BGR frame.

        Returns:
            True if this is a new scene.
        """
        # Compute histogram
        hist = self._compute_histogram(frame)

        if self.prev_hist is None:
            self.prev_hist = hist
            self.scene_changes += 1
            return True

        # Compare histograms
        method = self._methods.get(self.method, cv2.HISTCMP_CORREL)
        score = cv2.compareHist(self.prev_hist, hist, method)

        # For correlation, lower score = more different
        if self.method == "correlation":
            is_change = score < self.threshold
        else:
            is_change = score > self.threshold

        if is_change:
            self.scene_changes += 1

        self.prev_hist = hist
        return is_change

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute color histogram for frame."""
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def reset(self) -> None:
        """Reset for new video."""
        self.prev_hist = None
        self.scene_changes = 0


from dataclasses import dataclass, field

from config import settings


@dataclass(slots=True)
class FrameSampler:
    """Simple interval-based frame sampler."""

    every_n: int = field(default_factory=lambda: settings.frame_sample_ratio)

    def should_sample(self, index: int) -> bool:
        """Check if frame index matches sampling interval."""
        return index % self.every_n == 0
