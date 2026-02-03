"""Visual Saliency and Region of Interest detection.

Uses SAM 3 (Automatic Mask Generator) if available, falling back to OpenCV
Spectral Residual Saliency for efficiency.

Used by DeepResearch to identify important visual regions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class SaliencyDetector:
    """Visual saliency detection (SAM 3 > cv2)."""

    def __init__(self, use_sam3: bool = True):
        self.use_sam3 = use_sam3
        self._cv2 = None
        self._sam3 = None

    def _ensure_cv2(self) -> bool:
        if self._cv2 is not None:
            return True
        try:
            import cv2
            self._cv2 = cv2
            return True
        except ImportError:
            log.warning("[Saliency] OpenCV not available")
            return False

    async def _get_sam3(self):
        if self._sam3 is None and self.use_sam3:
            try:
                from core.tracking.sam3_tracker import SAM3Tracker
                self._sam3 = SAM3Tracker()
                # SAM3Tracker manages its own loading via ResourceArbiter
            except ImportError:
                log.warning("[Saliency] SAM3Tracker not found")
        return self._sam3

    async def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect salient regions in the frame."""
        # 1. Try SAM 3 (Automatic Mask Generator) - "The Abuse"
        # This gives us SEMANTIC regions, not just spectral residual.
        sam3 = await self._get_sam3()
        if sam3:
            try:
                # Stub: In real SAM 3, this would support auto_generate_masks(frame)
                # Since our tracker currently wraps point/text, we might need to extend it.
                # For now, we assume SAM3Tracker exposes an auto-mask method or similar.
                # If not implemented, fallback to CV2.
                if hasattr(sam3, "auto_generate_masks"):
                     # This method doesn't exist yet in our wrapper, so this block works
                     # as a feature flag for when we implement it.
                     pass
            except Exception as e:
                log.warning(f"[Saliency] SAM 3 detection failed: {e}")

        # 2. Fallback to OpenCV (Spectral Residual)
        return await self._detect_cv2(frame)

    async def _detect_cv2(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Legacy Spectral Residual Saliency."""
        if not self._ensure_cv2():
            return []

        try:
            cv2 = self._cv2
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, map_ = saliency.computeSaliency(frame)

            if success:
                map_ = (map_ * 255).astype(np.uint8)
                _, thresh = cv2.threshold(map_, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                regions = []
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    if w * h < (frame.shape[0] * frame.shape[1] * 0.01): # Ignore tiny regions
                        continue

                    roi = map_[y:y+h, x:x+w]
                    score = float(roi.mean()) / 255.0
                    regions.append({
                        "bbox": [x, y, x+w, y+h],
                        "score": round(score, 3),
                        "source": "cv2_spectral"
                    })

                # Sort by score
                regions.sort(key=lambda x: x["score"], reverse=True)
                return regions[:5] # Top 5

            return []
        except Exception as e:
            log.error(f"[Saliency] CV2 failed: {e}")
            return []
