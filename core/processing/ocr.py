"""OCR pipeline for extracting text from video frames.

Uses PaddleOCR for multilingual text extraction from frames,
enabling search for on-screen text like signs, titles, and captions.

Supported OCR Engines (The Council):
- PaddleOCR (Primary): 80+ languages, scene text master
- EasyOCR (Alternative): Simple setup, good quality
- Surya (Documents): Best for slides/PDFs/presentations
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


# === MULTILINGUAL SUPPORT (80+ Languages) ===
# PaddleOCR language codes for global coverage
PADDLEOCR_LANGUAGES = {
    # Latin Script
    "en": "en",  # English
    "fr": "fr",  # French
    "de": "german",  # German
    "es": "es",  # Spanish
    "pt": "pt",  # Portuguese
    "it": "it",  # Italian
    "nl": "nl",  # Dutch
    "pl": "pl",  # Polish
    "ro": "ro",  # Romanian
    "tr": "tr",  # Turkish
    "vi": "vi",  # Vietnamese
    "id": "id",  # Indonesian
    "ms": "ms",  # Malay
    # South Asian
    "hi": "hi",  # Hindi
    "ta": "ta",  # Tamil
    "te": "te",  # Telugu
    "kn": "kn",  # Kannada
    "mr": "mr",  # Marathi
    "ne": "ne",  # Nepali
    "bn": "bn",  # Bengali
    "gu": "gu",  # Gujarati
    # East Asian
    "ch": "ch",  # Chinese (Simplified)
    "cht": "cht",  # Chinese (Traditional)
    "ja": "japan",  # Japanese
    "ko": "korean",  # Korean
    # Middle Eastern
    "ar": "ar",  # Arabic
    "fa": "fa",  # Persian/Farsi
    "ur": "ur",  # Urdu
    "he": "he",  # Hebrew
    # Cyrillic
    "ru": "ru",  # Russian
    "uk": "uk",  # Ukrainian
    "bg": "bg",  # Bulgarian
    "sr": "rs_cyrillic",  # Serbian
    # Other
    "th": "th",  # Thai
    "el": "el",  # Greek
}


class OCRProcessor:
    """Extract text from video frames using PaddleOCR.

    Supports 80+ languages via PaddleOCR's multilingual models.

    Usage:
        ocr = OCRProcessor()
        result = await ocr.extract_text(frame)
        # {"text": "WELCOME TO...", "boxes": [...], "confidence": 0.95}

        # Multilingual
        ocr = OCRProcessor(lang="ta")  # Tamil
        ocr = OCRProcessor(lang="multilingual")  # Auto-detect
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        enable_angle_cls: bool = True,
    ):
        """Initialize OCR processor.

        Args:
            lang: Language code ('en', 'ta', 'hi', 'ch', 'multilingual', etc.)
            use_gpu: Whether to use GPU acceleration.
            enable_angle_cls: Enable text angle classification.
        """
        # Map to PaddleOCR lang code
        self.lang = PADDLEOCR_LANGUAGES.get(lang, lang)
        self.use_gpu = use_gpu
        self.enable_angle_cls = enable_angle_cls
        self.ocr = None
        self._init_lock = asyncio.Lock()

        # For multilingual mode, we'll use Chinese model which includes English
        if lang == "multilingual":
            self.lang = "ch"  # Chinese model handles mixed scripts best

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
                    use_angle_cls=self.enable_angle_cls,
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
            # Run OCR in a thread to prevent blocking the event loop
            result = await asyncio.to_thread(self.ocr.ocr, frame, cls=True)

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


class SuryaOCR:
    """High-quality document OCR using Surya (by VikP).

    Best for: PDF slides, presentations, printed documents.
    NOT recommended for: Scene text (street signs, etc.)

    Usage:
        ocr = SuryaOCR()
        result = await ocr.extract_text(frame)
    """

    def __init__(self, use_gpu: bool = True):
        """Initialize Surya OCR for document text.

        Args:
            use_gpu: Whether to use GPU acceleration.
        """
        self.use_gpu = use_gpu
        self._detector = None
        self._recognizer = None
        self._init_lock = asyncio.Lock()

    async def _lazy_load(self) -> bool:
        """Load Surya models lazily."""
        if self._detector is not None and self._recognizer is not None:
            return True

        async with self._init_lock:
            if self._detector is not None and self._recognizer is not None:
                return True

            try:
                from surya.detection import batch_detection
                from surya.model.detection import load_detector
                from surya.model.recognition import load_recognizer
                from surya.recognition import batch_recognition

                log.info("[SuryaOCR] Loading models...")
                # Load models in threads to avoid blocking the event loop
                self._detector = await asyncio.to_thread(load_detector)
                self._recognizer = await asyncio.to_thread(load_recognizer)
                self._batch_detection = batch_detection
                self._batch_recognition = batch_recognition
                log.info("[SuryaOCR] Models loaded")
                return True

            except ImportError:
                log.warning(
                    "[SuryaOCR] surya-ocr not installed (pip install surya-ocr)"
                )
                return False
            except Exception as e:
                log.error(f"[SuryaOCR] Failed to load: {e}")
                return False

    async def extract_text(
        self,
        frame: np.ndarray,
        min_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Extract text from document/slide frame.

        Args:
            frame: RGB frame as numpy array.
            min_confidence: Minimum confidence threshold.

        Returns:
            Dict with 'text', 'boxes', 'lines', and 'confidence'.
        """
        if not await self._lazy_load():
            return {"text": "", "boxes": [], "lines": [], "confidence": 0.0}

        try:
            from PIL import Image

            # Convert numpy to PIL
            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame

            # Use GPU_SEMAPHORE if available to manage VRAM
            # (Checking for GPU_SEMAPHORE in globals)
            sem = globals().get("GPU_SEMAPHORE", asyncio.Semaphore(1))

            async with sem:
                # Detect text regions off-thread
                det_results = await asyncio.to_thread(self._batch_detection, [image], self._detector)

                # Recognize text off-thread
                rec_results = await asyncio.to_thread(self._batch_recognition, 
                    [image], det_results, self._recognizer
                )

            if not rec_results or not rec_results[0]:
                return {"text": "", "boxes": [], "lines": [], "confidence": 0.0}

            lines = []
            boxes = []
            confidences = []

            for line in rec_results[0].text_lines:
                text = line.text
                conf = line.confidence

                if conf >= min_confidence and text.strip():
                    lines.append(text)
                    boxes.append(line.bbox)
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
            log.error(f"[SuryaOCR] Extraction failed: {e}")
            return {"text": "", "boxes": [], "lines": [], "confidence": 0.0}

    def cleanup(self) -> None:
        """Release Surya resources."""
        self._detector = None
        self._recognizer = None
        log.info("[SuryaOCR] Resources released")


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
            # Run EasyOCR in a thread
            result = await asyncio.to_thread(self.reader.readtext, frame)

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
