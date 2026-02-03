"""Pluggable OCR Factory for selecting the best available OCR engine.

Allows switching between PaddleOCR (recommended, 80+ languages), EasyOCR,
and potentially others like Surya without changing pipeline code.
"""

from __future__ import annotations

from typing import Any

from config import settings
from core.utils.logger import get_logger

log = get_logger(__name__)


def get_ocr_engine() -> Any:
    """Factory function to get the configured OCR engine instance.

    Read 'ocr_engine' from settings (default: 'paddle').
    Options:
        - 'paddle': PaddleOCR (Multi-language, Fast, Accurate)
        - 'easy': EasyOCR (Pure Python, Easy setup)
        - 'surya': SuryaOCR (Document specialized)

    Returns:
        An initialized OCR processor instance conforming to the extract_text interface.
    """
    engine_type = getattr(settings, "ocr_engine", "paddle").lower()
    lang = getattr(settings, "ocr_language", "multilingual")

    log.info(f"[OCR] Initializing engine: {engine_type} (lang={lang})")

    try:
        if engine_type == "paddle":
            from core.processing.ocr import OCRProcessor
            return OCRProcessor(lang=lang, use_gpu=True)

        elif engine_type == "easy":
            from core.processing.ocr import EasyOCRProcessor
            # EasyOCR expects list of languages
            langs = [lang] if lang != "multilingual" else ["en"]
            return EasyOCRProcessor(langs=langs, use_gpu=True)

        elif engine_type == "surya":
            # Placeholder for Surya integration
            try:
                from core.processing.ocr import SuryaOCRProcessor
                return SuryaOCRProcessor()
            except ImportError:
                log.warning("SuryaOCR not found, falling back to PaddleOCR")
                from core.processing.ocr import OCRProcessor
                return OCRProcessor(lang=lang, use_gpu=True)

        else:
            log.warning(f"Unknown OCR engine '{engine_type}', defaulting to PaddleOCR")
            from core.processing.ocr import OCRProcessor
            return OCRProcessor(lang=lang, use_gpu=True)

    except Exception as e:
        log.error(f"[OCR] Failed to initialize {engine_type}: {e}")
        # Last resort fallback to EasyOCR if Paddle fails (easier dependency)
        if engine_type != "easy":
            log.info("[OCR] Falling back to EasyOCR")
            try:
                from core.processing.ocr import EasyOCRProcessor
                return EasyOCRProcessor(langs=["en"], use_gpu=True)
            except Exception:
                pass
        raise e
