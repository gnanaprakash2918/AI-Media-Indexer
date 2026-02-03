"""Dependency checker to fail fast if critical models are missing."""

import importlib
import logging

from config import settings

logger = logging.getLogger(__name__)


def check_model_dependencies():
    """Checks if all required packages for enabled features are installed.

    Raises:
        ImportError: If a critical dependency is missing.
    """
    missing = []

    # 1. OCR Dependencies
    if settings.enable_ocr:
        if settings.ocr_engine == "paddle":
            if not importlib.util.find_spec("paddleocr"):
                missing.append("paddleocr (required for PaddleOCR)")
            if not importlib.util.find_spec("paddle"):
                missing.append("paddlepaddle (required for PaddleOCR)")
        elif settings.ocr_engine == "easy":
            if not importlib.util.find_spec("easyocr"):
                missing.append("easyocr (required for EasyOCR)")

    # 2. Deep Research / Video Understanding
    if settings.enable_video_embeddings:
        if settings.enable_internvideo:
            if not importlib.util.find_spec("timm"):
                missing.append("timm (required for InternVideo)")
            # InternVideo usually requires Decord for video reading internally
            if not importlib.util.find_spec("decord"):
                missing.append("decord (required for high-perf video reading)")

    # 3. Audio / Voice
    if not importlib.util.find_spec("librosa"):
        missing.append("librosa (required for audio analysis)")

    # 4. SAM3 (if we ever re-enable it or for future object tracking)
    # This is just a check, currently SAM3 is disabled in defaults
    # if not importlib.util.find_spec("sam2"):
    #     logger.warning("SAM2 not found - Object Tracking will be disabled.")

    if missing:
        raise ImportError(
            f"Missing critical dependencies for enabled features:\n"
            f"{chr(10).join(['- ' + m for m in missing])}\n"
            f"Please run: pip install {' '.join([m.split()[0] for m in missing])}"
        )

    logger.info("Dependency check passed: All critical models are loadable.")
