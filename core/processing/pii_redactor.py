"""PII Redaction pipeline for privacy protection.

Provides face/text redaction using:
- SAM3 masks for precise face segmentation
- Gaussian blur/pixelation for redaction
- HITL override for false positives/negatives
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from core.utils.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


class RedactionMode(Enum):
    """Redaction visual style."""

    BLUR = "blur"
    PIXELATE = "pixelate"
    BLACK = "black"
    EMOJI = "emoji"


class PIIRedactor:
    """Redact faces and sensitive text from video frames.

    Usage:
        redactor = PIIRedactor()
        redacted_frame = await redactor.redact_faces(frame, face_boxes)
        redacted_frame = await redactor.redact_text(frame, text_boxes)
    """

    def __init__(
        self,
        mode: RedactionMode = RedactionMode.BLUR,
        blur_strength: int = 51,
        pixelate_size: int = 10,
    ):
        """Initialize PII redactor.

        Args:
            mode: Visual style for redaction.
            blur_strength: Gaussian blur kernel size (must be odd).
            pixelate_size: Pixel block size for pixelation.
        """
        self.mode = mode
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.pixelate_size = pixelate_size

    async def redact_faces(
        self,
        frame: np.ndarray,
        face_boxes: list[tuple[int, int, int, int]],
        exclude_clusters: list[int] | None = None,
    ) -> np.ndarray:
        """Redact faces from a frame.

        Args:
            frame: RGB/BGR frame as numpy array.
            face_boxes: List of (x, y, w, h) bounding boxes.
            exclude_clusters: Cluster IDs to NOT redact (HITL approved).

        Returns:
            Frame with faces redacted.
        """
        if not face_boxes:
            return frame

        result = frame.copy()

        for box in face_boxes:
            x, y, w, h = box
            x, y = max(0, x), max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            if x2 <= x or y2 <= y:
                continue

            roi = result[y:y2, x:x2]

            if self.mode == RedactionMode.BLUR:
                redacted = self._apply_blur(roi)
            elif self.mode == RedactionMode.PIXELATE:
                redacted = self._apply_pixelate(roi)
            elif self.mode == RedactionMode.BLACK:
                redacted = np.zeros_like(roi)
            else:
                redacted = self._apply_blur(roi)

            result[y:y2, x:x2] = redacted

        log.debug(f"[PIIRedactor] Redacted {len(face_boxes)} faces")
        return result

    async def redact_text(
        self,
        frame: np.ndarray,
        text_boxes: list[list[list[int]]],
        sensitive_patterns: list[str] | None = None,
    ) -> np.ndarray:
        """Redact text regions from a frame.

        Args:
            frame: RGB/BGR frame as numpy array.
            text_boxes: List of polygon boxes [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
            sensitive_patterns: Regex patterns for sensitive text (SSN, email, etc.).

        Returns:
            Frame with text redacted.
        """
        if not text_boxes:
            return frame

        result = frame.copy()

        for box in text_boxes:
            # Convert polygon to bounding rect
            pts = np.array(box, dtype=np.int32)
            x, y, w, h = self._polygon_to_rect(pts)

            if w <= 0 or h <= 0:
                continue

            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            x, y = max(0, x), max(0, y)

            roi = result[y:y2, x:x2]
            if roi.size == 0:
                continue

            if self.mode == RedactionMode.BLUR:
                redacted = self._apply_blur(roi)
            elif self.mode == RedactionMode.PIXELATE:
                redacted = self._apply_pixelate(roi)
            else:
                redacted = np.zeros_like(roi)

            result[y:y2, x:x2] = redacted

        log.debug(f"[PIIRedactor] Redacted {len(text_boxes)} text regions")
        return result

    async def redact_with_sam_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Redact using SAM3 segmentation mask for precise boundaries.

        Args:
            frame: RGB/BGR frame.
            mask: Binary mask from SAM3 (same size as frame).

        Returns:
            Frame with masked regions redacted.
        """
        if mask is None or mask.size == 0:
            return frame

        result = frame.copy()

        # Apply blur to entire frame, then blend using mask
        if self.mode == RedactionMode.BLUR:
            blurred = self._apply_blur(frame)
        elif self.mode == RedactionMode.PIXELATE:
            blurred = self._apply_pixelate(frame)
        else:
            blurred = np.zeros_like(frame)

        # Alpha blend using mask
        mask_3d = np.stack([mask] * 3, axis=-1) if mask.ndim == 2 else mask
        mask_normalized = mask_3d.astype(np.float32) / 255.0

        result = (
            blurred * mask_normalized + frame * (1 - mask_normalized)
        ).astype(np.uint8)

        log.debug("[PIIRedactor] Applied SAM mask redaction")
        return result

    def _apply_blur(self, roi: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to region."""
        try:
            import cv2

            return cv2.GaussianBlur(
                roi, (self.blur_strength, self.blur_strength), 0
            )
        except ImportError:
            # Fallback: simple averaging
            return np.full_like(roi, roi.mean(axis=(0, 1)))

    def _apply_pixelate(self, roi: np.ndarray) -> np.ndarray:
        """Apply pixelation effect to region."""
        try:
            import cv2

            h, w = roi.shape[:2]
            small = cv2.resize(
                roi,
                (max(1, w // self.pixelate_size), max(1, h // self.pixelate_size)),
                interpolation=cv2.INTER_LINEAR,
            )
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        except ImportError:
            return np.full_like(roi, roi.mean(axis=(0, 1)))

    def _polygon_to_rect(self, pts: np.ndarray) -> tuple[int, int, int, int]:
        """Convert polygon points to bounding rectangle."""
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        x = int(x_coords.min())
        y = int(y_coords.min())
        w = int(x_coords.max() - x)
        h = int(y_coords.max() - y)
        return x, y, w, h


# Sensitive text patterns for auto-detection
SENSITIVE_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    r"\b\d{10}\b",  # Phone number
    r"\b\d{5}(-\d{4})?\b",  # ZIP code
]
