"""Integration hooks for new AI-Media-Indexer capabilities.

Provides clean hooks to enable the new modules in the existing pipeline:
- ResourceArbiter for VRAM management
- HybridSearcher for BM25+vector search
- SmartFrameSampler for VLM optimization
- AudioEventDetector for CLAP
- OCR and object detection
- Temporal action recognition

Usage:
    from core.integration import get_enhanced_pipeline_config
    config = get_enhanced_pipeline_config()
    # Use config to enable new features in pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.utils.logger import get_logger

if TYPE_CHECKING:
    from core.processing.audio_events import AudioEventDetector
    from core.processing.biometric_arbitrator import BiometricArbitrator
    from core.processing.frame_sampling import SmartFrameSampler, TextGatedOCR
    from core.processing.object_detection import ObjectDetector
    from core.processing.ocr import OCRProcessor
    from core.processing.temporal import TemporalAnalyzer
    from core.retrieval.hybrid import HybridSearcher
    from core.retrieval.privacy import PrivacyFilter
    from core.storage.keyword_index import KeywordIndex
    from core.utils.cancellation import CancellationToken
    from core.utils.resource_arbiter import ResourceArbiter

log = get_logger(__name__)





def get_resource_arbiter() -> ResourceArbiter:
    """Get the global ResourceArbiter for VRAM management."""
    from core.utils.resource_arbiter import RESOURCE_ARBITER

    return RESOURCE_ARBITER


def get_cancellation_token(job_id: str) -> CancellationToken:
    """Get or create a cancellation token for a job."""
    from core.utils.cancellation import get_or_create_token

    return get_or_create_token(job_id)


def cancel_job(job_id: str) -> bool:
    """Cancel a running job."""
    from core.utils.cancellation import cancel_job as _cancel

    return _cancel(job_id)


def safe_path(path_str: str | Path) -> Path:
    """Get a Windows-safe path."""
    from core.utils.filesystem import safe_path as _safe

    return _safe(path_str)


def get_biometric_arbitrator() -> BiometricArbitrator:
    """Get the global BiometricArbitrator for face verification."""
    from core.processing.biometric_arbitrator import BIOMETRIC_ARBITRATOR

    return BIOMETRIC_ARBITRATOR


def get_privacy_filter() -> PrivacyFilter:
    """Get the global PrivacyFilter for personal/movie mode."""
    from core.retrieval.privacy import PRIVACY_FILTER

    return PRIVACY_FILTER
