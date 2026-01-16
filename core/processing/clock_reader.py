"""Module for reading digital clocks from video frames."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class ClockReader:
    """Reads digital clocks from images using EasyOCR."""

    def __init__(self):
        """Initialize OCR processor resources."""
        self._ocr = None
        self._init_lock = asyncio.Lock()

    async def _ensure_ocr(self) -> bool:
        if self._ocr is not None:
            return True

        async with self._init_lock:
            if self._ocr is not None:
                return True
            try:
                from core.processing.ocr import EasyOCRProcessor

                self._ocr = EasyOCRProcessor(langs=["en"], use_gpu=True)
                log.info("[ClockReader] OCR initialized")
                return True
            except Exception as e:
                log.error(f"[ClockReader] OCR init failed: {e}")
                return False

    async def read_digital_clock(
        self, image: np.ndarray | Path
    ) -> dict[str, Any]:
        """Detect and read digital clock time from image."""
        if not await self._ensure_ocr() or self._ocr is None:
            return {"time": None, "error": "OCR not loaded"}

        try:
            from PIL import Image

            if isinstance(image, Path):
                img = np.array(Image.open(image))
            elif not isinstance(image, np.ndarray):
                img = np.array(image)
            else:
                img = image

            result = await self._ocr.extract_text(img)
            raw_text = result.get("text", "")

            time_patterns = [
                r"(\d{1,2}):(\d{2}):(\d{2})",
                r"(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?",
                r"(\d{1,2})\.(\d{2})\s*(AM|PM|am|pm)?",
                r"(\d{4})",
            ]

            for pattern in time_patterns:
                match = re.search(pattern, raw_text)
                if match:
                    time_str = match.group(0)
                    log.info(f"[ClockReader] Digital time detected: {time_str}")
                    return {
                        "time": time_str,
                        "format": "digital",
                        "raw_ocr": raw_text,
                        "confidence": result.get("confidence", 0.0),
                    }

            return {
                "time": None,
                "format": "digital",
                "raw_ocr": raw_text,
                "error": "No time pattern found",
            }

        except Exception as e:
            log.error(f"[ClockReader] Digital read failed: {e}")
            return {"time": None, "error": str(e)}

    async def read_analog_clock(
        self, image: np.ndarray | Path
    ) -> dict[str, Any]:
        """Detect and read analog clock time from image."""
        try:
            import cv2
            from PIL import Image

            if isinstance(image, Path):
                img = np.array(Image.open(image))
            elif not isinstance(image, np.ndarray):
                img = np.array(image)
            else:
                img = image

            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=min(gray.shape) // 2,
            )

            if circles is None:
                return {
                    "time": None,
                    "format": "analog",
                    "error": "No clock face detected",
                }

            circle = circles[0][0]
            cx, cy, radius = int(circle[0]), int(circle[1]), int(circle[2])

            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=30,
                minLineLength=int(radius * 0.3),
                maxLineGap=10,
            )

            if lines is None:
                return {
                    "time": None,
                    "format": "analog",
                    "error": "No hands detected",
                }

            hands = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dist_to_center = min(
                    np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2),
                    np.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2),
                )
                if dist_to_center < radius * 0.3:
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    hands.append(
                        {
                            "length": length,
                            "angle": angle,
                            "coords": (x1, y1, x2, y2),
                        }
                    )

            hands.sort(key=lambda h: h["length"], reverse=True)

            if len(hands) >= 2:
                minute_hand = hands[0]
                hour_hand = hands[1]

                minute_angle = (minute_hand["angle"] + 90) % 360
                hour_angle = (hour_hand["angle"] + 90) % 360

                minutes = int((minute_angle / 360) * 60) % 60
                hours = int((hour_angle / 360) * 12) % 12
                if hours == 0:
                    hours = 12

                time_str = f"{hours:02d}:{minutes:02d}"
                log.info(f"[ClockReader] Analog time detected: {time_str}")
                return {
                    "time": time_str,
                    "format": "analog",
                    "hours": hours,
                    "minutes": minutes,
                    "confidence": 0.6,
                }

            return {
                "time": None,
                "format": "analog",
                "error": "Insufficient hands detected",
            }

        except Exception as e:
            log.error(f"[ClockReader] Analog read failed: {e}")
            return {"time": None, "error": str(e)}

    async def read_time(self, image: np.ndarray | Path) -> dict[str, Any]:
        """Detect and read time (digital or analog) from image."""
        digital_result = await self.read_digital_clock(image)
        if digital_result.get("time"):
            return digital_result

        analog_result = await self.read_analog_clock(image)
        if analog_result.get("time"):
            return analog_result

        return {
            "time": None,
            "error": "No clock detected (tried digital and analog)",
        }

    def cleanup(self) -> None:
        """Clear OCR resources."""
        self._ocr = None
        log.info("[ClockReader] Resources released")

    async def matches_time_constraint(
        self,
        image: np.ndarray | Path,
        min_time: str | None = None,
        max_time: str | None = None,
    ) -> dict[str, Any]:
        """Check if the time visible in the image matches constraints.

        Args:
            image: Input image.
            min_time: Minimum allowed time (HH:MM:SS or HH:MM).
            max_time: Maximum allowed time (HH:MM:SS or HH:MM).

        Returns:
            Match result.
        """
        result = await self.read_time(image)
        if result.get("error"):
            result["matches"] = False
            return result

        time_str = result.get("time", "")
        matches = True

        if min_time or max_time:
            # Simple lexicographical comparison for HH:MM:SS format
            if min_time and time_str < min_time:
                matches = False
            if max_time and time_str > max_time:
                matches = False

        result["matches"] = matches
        return result
