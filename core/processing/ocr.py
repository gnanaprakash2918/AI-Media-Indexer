"""OCR pipeline for extracting text from video frames.

Uses PaddleOCR for multilingual text extraction from frames,
enabling search for on-screen text like signs, titles, and captions.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class OCRProcessor:
    """Extract text from video frames using PaddleOCR.

    Usage:
        ocr = OCRProcessor()
        result = await ocr.extract_text(frame)
        # {"text": "WELCOME TO...", "boxes": [...], "confidence": 0.95}
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
    ):
        """Initialize OCR processor.

        Args:
            lang: Language code ('en', 'ta', 'hi', 'ch', etc.)
            use_gpu: Whether to use GPU acceleration.
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr = None
        self._init_lock = asyncio.Lock()

    async def _lazy_load(self) -> bool:
        """Load PaddleOCR model lazily.

        Returns:
            True if model loaded successfully.
        """
        if self.ocr is not None:
            return True

        async with self._init_lock:
            if self.ocr is not None:
                return True

            try:
                from paddleocr import PaddleOCR

                log.info(f"[OCR] Loading PaddleOCR lang={self.lang}")
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=False,
                )
                log.info("[OCR] Model loaded")
                return True

            except ImportError:
                log.warning("[OCR] paddleocr not installed")
                return False
            except Exception as e:
                log.error(f"[OCR] Failed to load: {e}")
                return False

    async def extract_text(
        self,
        frame: np.ndarray,
        min_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Extract text from a frame.

        Args:
            frame: RGB/BGR frame as numpy array.
            min_confidence: Minimum confidence threshold.

        Returns:
            Dict with 'text', 'boxes', 'lines', and 'confidence'.
        """
        if not await self._lazy_load():
            return {"text": "", "boxes": [], "lines": [], "confidence": 0.0}

        if self.ocr is None:
            return {"text": "", "boxes": [], "lines": [], "confidence": 0.0}

        try:
            result = self.ocr.ocr(frame, cls=True)

            if not result or not result[0]:
                return {
                    "text": "",
                    "boxes": [],
                    "lines": [],
                    "confidence": 0.0,
                }

            lines = []
            boxes = []
            confidences = []

            for line in result[0]:
                box = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = line[1][0]
                if not text or len(text) < 2:
                    continue
                # Cast confidence to float for comparison
                try:
                    conf = float(line[1][1])
                except (ValueError, IndexError, TypeError):
                    conf = 0.0

                if conf >= min_confidence:
                    lines.append(text)
                    boxes.append(box)
                    confidences.append(conf)

            full_text = " ".join(lines)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            return {
                "text": full_text,
                "boxes": boxes,
                "lines": lines,
                "confidence": round(avg_conf, 3),
            }

        except Exception as e:
            log.error(f"[OCR] Extraction failed: {e}")
            return {"text": "", "boxes": [], "lines": [], "confidence": 0.0}

    async def extract_batch(
        self,
        frames: list[np.ndarray],
        min_confidence: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Extract text from multiple frames.

        Args:
            frames: List of frames.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of extraction results.
        """
        results = []
        for frame in frames:
            result = await self.extract_text(frame, min_confidence)
            results.append(result)
        return results

    def cleanup(self) -> None:
        """Release OCR resources."""
        if self.ocr:
            del self.ocr
            self.ocr = None
        log.info("[OCR] Resources released")


class EasyOCRProcessor:
    """Alternative OCR using EasyOCR (simpler setup).

    Usage:
        ocr = EasyOCRProcessor(langs=['en', 'ta'])
        text = await ocr.extract_text(frame)
    """

    def __init__(self, langs: list[str] | None = None, use_gpu: bool = True):
        """Initialize EasyOCR processor.

        Args:
            langs: List of language codes. Defaults to English.
            use_gpu: Whether to use GPU.
        """
        self.langs = langs or ["en"]
        self.use_gpu = use_gpu
        self.reader = None
        self._init_lock = asyncio.Lock()

    async def _lazy_load(self) -> bool:
        """Load EasyOCR model lazily."""
        if self.reader is not None:
            return True

        async with self._init_lock:
            if self.reader is not None:
                return True

            try:
                import easyocr

                log.info(f"[EasyOCR] Loading langs={self.langs}")
                self.reader = easyocr.Reader(
                    self.langs,
                    gpu=self.use_gpu,
                )
                log.info("[EasyOCR] Model loaded")
                return True

            except ImportError:
                log.warning("[EasyOCR] easyocr not installed")
                return False
            except Exception as e:
                log.error(f"[EasyOCR] Failed to load: {e}")
                return False

    async def extract_text(
        self,
        frame: np.ndarray,
        min_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Extract text from a frame."""
        if not await self._lazy_load():
            return {"text": "", "boxes": [], "confidence": 0.0}

        if self.reader is None:
            return {"text": "", "boxes": [], "confidence": 0.0}

        try:
            result = self.reader.readtext(frame)

            lines = []
            boxes = []
            confidences = []

            for bbox, text, conf in result:
                # Cast confidence to float for comparison
                try:
                    conf_val = float(conf)
                except (ValueError, TypeError):
                    conf_val = 0.0

                if conf_val >= min_confidence:
                    lines.append(text)
                    boxes.append(bbox)
                    confidences.append(conf_val)

            full_text = " ".join(lines)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            return {
                "text": full_text,
                "boxes": boxes,
                "lines": lines,
                "confidence": round(avg_conf, 3),
            }

        except Exception as e:
            log.error(f"[EasyOCR] Extraction failed: {e}")
            return {"text": "", "boxes": [], "confidence": 0.0}

    def cleanup(self) -> None:
        """Release resources."""
        if self.reader:
            del self.reader
            self.reader = None
        log.info("[EasyOCR] Resources released")
