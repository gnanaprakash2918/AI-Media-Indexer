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
    from core.processing.frame_sampling import SmartFrameSampler, TextGatedOCR
    from core.processing.ocr import OCRProcessor
    from core.processing.object_detection import ObjectDetector
    from core.processing.temporal import TemporalAnalyzer
    from core.retrieval.hybrid import HybridSearcher
    from core.storage.keyword_index import KeywordIndex
    from core.utils.cancellation import CancellationToken
    from core.utils.resource_arbiter import ResourceArbiter

log = get_logger(__name__)


class EnhancedPipelineConfig:
    """Configuration for enhanced pipeline capabilities.

    Centralizes initialization of all new modules for easy integration
    into the existing IngestionPipeline.
    """

    def __init__(
        self,
        enable_smart_sampling: bool = True,
        enable_clap: bool = True,
        enable_ocr: bool = True,
        enable_object_detection: bool = False,
        enable_temporal: bool = False,
        enable_hybrid_search: bool = True,
        bm25_path: Path | None = None,
    ):
        """Initialize enhanced pipeline configuration.

        Args:
            enable_smart_sampling: Enable motion-gated frame sampling.
            enable_clap: Enable CLAP audio event detection.
            enable_ocr: Enable OCR text extraction.
            enable_object_detection: Enable YOLO-World (requires ultralytics).
            enable_temporal: Enable TimeSformer action recognition.
            enable_hybrid_search: Enable BM25 + vector hybrid search.
            bm25_path: Path for BM25 index persistence.
        """
        self.enable_smart_sampling = enable_smart_sampling
        self.enable_clap = enable_clap
        self.enable_ocr = enable_ocr
        self.enable_object_detection = enable_object_detection
        self.enable_temporal = enable_temporal
        self.enable_hybrid_search = enable_hybrid_search
        self.bm25_path = bm25_path or Path("data/bm25_index.pkl")

        # Lazy-loaded components
        self._frame_sampler: SmartFrameSampler | None = None
        self._text_gate: TextGatedOCR | None = None
        self._audio_detector: AudioEventDetector | None = None
        self._ocr: OCRProcessor | None = None
        self._object_detector: ObjectDetector | None = None
        self._temporal: TemporalAnalyzer | None = None
        self._hybrid_searcher: HybridSearcher | None = None
        self._keyword_index: KeywordIndex | None = None

    @property
    def frame_sampler(self) -> SmartFrameSampler | None:
        """Get motion-gated frame sampler."""
        if not self.enable_smart_sampling:
            return None
        if self._frame_sampler is None:
            from core.processing.frame_sampling import SmartFrameSampler
            self._frame_sampler = SmartFrameSampler(
                motion_threshold=30.0,
                min_analyze_interval=5,
            )
            log.info("[Integration] SmartFrameSampler enabled")
        return self._frame_sampler

    @property
    def text_gate(self) -> TextGatedOCR | None:
        """Get text detection gate for OCR optimization."""
        if not self.enable_ocr:
            return None
        if self._text_gate is None:
            from core.processing.frame_sampling import TextGatedOCR
            self._text_gate = TextGatedOCR(edge_threshold=0.02)
            log.info("[Integration] TextGatedOCR enabled")
        return self._text_gate

    @property
    def audio_detector(self) -> AudioEventDetector | None:
        """Get CLAP audio event detector."""
        if not self.enable_clap:
            return None
        if self._audio_detector is None:
            from core.processing.audio_events import AudioEventDetector
            self._audio_detector = AudioEventDetector()
            log.info("[Integration] CLAP AudioEventDetector enabled")
        return self._audio_detector

    @property
    def ocr(self) -> OCRProcessor | None:
        """Get OCR processor."""
        if not self.enable_ocr:
            return None
        if self._ocr is None:
            from core.processing.ocr import OCRProcessor
            self._ocr = OCRProcessor(lang="en", use_gpu=True)
            log.info("[Integration] OCRProcessor enabled")
        return self._ocr

    @property
    def object_detector(self) -> ObjectDetector | None:
        """Get YOLO-World object detector."""
        if not self.enable_object_detection:
            return None
        if self._object_detector is None:
            from core.processing.object_detection import ObjectDetector
            self._object_detector = ObjectDetector(model_size="m")
            log.info("[Integration] YOLO-World ObjectDetector enabled")
        return self._object_detector

    @property
    def temporal(self) -> TemporalAnalyzer | None:
        """Get TimeSformer temporal analyzer."""
        if not self.enable_temporal:
            return None
        if self._temporal is None:
            from core.processing.temporal import TemporalAnalyzer
            self._temporal = TemporalAnalyzer()
            log.info("[Integration] TimeSformer TemporalAnalyzer enabled")
        return self._temporal

    def get_hybrid_searcher(self, db: Any) -> HybridSearcher | None:
        """Get hybrid searcher with BM25 + vector search.

        Args:
            db: VectorDB instance for hybrid search.

        Returns:
            HybridSearcher instance if enabled.
        """
        if not self.enable_hybrid_search:
            return None
        if self._hybrid_searcher is None:
            from core.retrieval.hybrid import HybridSearcher
            self._hybrid_searcher = HybridSearcher(db, self.bm25_path)
            log.info("[Integration] HybridSearcher enabled")
        return self._hybrid_searcher

    def get_keyword_index(self) -> KeywordIndex | None:
        """Get standalone BM25 keyword index."""
        if self._keyword_index is None:
            from core.storage.keyword_index import KeywordIndex
            self._keyword_index = KeywordIndex(self.bm25_path)
            self._keyword_index.load()
            log.info("[Integration] KeywordIndex enabled")
        return self._keyword_index

    def cleanup(self) -> None:
        """Release all resources."""
        if self._audio_detector:
            self._audio_detector.cleanup()
        if self._ocr:
            self._ocr.cleanup()
        if self._object_detector:
            self._object_detector.cleanup()
        if self._temporal:
            self._temporal.cleanup()
        if self._keyword_index:
            self._keyword_index.save()
        log.info("[Integration] All resources released")


# Global configuration instance
_enhanced_config: EnhancedPipelineConfig | None = None


def get_enhanced_config() -> EnhancedPipelineConfig:
    """Get or create the global enhanced pipeline configuration."""
    global _enhanced_config
    if _enhanced_config is None:
        _enhanced_config = EnhancedPipelineConfig()
    return _enhanced_config


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
