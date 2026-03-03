"""Deep Research Integration Module.

Wires SOTA models into the ingestion pipeline:
- Cinematography (DynamicClassifier, AestheticScorer, TechnicalCueDetector)
- Video Understanding (InternVideo2, LanguageBind) — gated by config

User Priority: ACCURACY over storage/speed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DeepResearchResult:
    """Container for all Deep Research analysis results."""

    # Video understanding
    video_features: dict[str, np.ndarray] = field(default_factory=dict)
    action_labels: list[str] = field(default_factory=list)

    # Cinematography
    shot_type: str = ""
    shot_confidence: float = 0.0
    aesthetic_score: float = 0.0
    mood: str = ""
    mood_confidence: float = 0.0

    # Technical cues
    is_black_frame: bool = False
    is_shot_boundary: bool = False


class DeepResearchProcessor:
    """Unified processor for Deep Research SOTA models.

    Provides a single interface to run cinematography + video understanding
    models on frames/video. All models are lazy-loaded.

    Usage:
        processor = DeepResearchProcessor()
        result = await processor.analyze_frame(frame, ...)
        result = await processor.analyze_video_segment(video_path, start, end)
    """

    def __init__(
        self,
        enable_video_understanding: bool = True,
        enable_cinematography: bool = True,
        device: str | None = None,
    ):
        self._enable_video = enable_video_understanding
        self._enable_cinematography = enable_cinematography
        self._device = device

        # Lazy-loaded components
        self._dynamic_classifier = None
        self._aesthetic_scorer = None
        self._technical_detector = None
        self._languagebind = None
        self._internvideo = None

        self._init_lock = asyncio.Lock()

    # =========================================================================
    # LAZY LOADERS
    # =========================================================================

    async def _get_dynamic_classifier(self):
        """Lazy load DynamicClassifier for shot types, moods, etc."""
        if self._dynamic_classifier is None and self._enable_cinematography:
            try:
                from core.processing.cinematography import DynamicClassifier

                self._dynamic_classifier = DynamicClassifier(
                    device=self._device
                )
                log.info("[DeepResearch] DynamicClassifier loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] DynamicClassifier failed: {e}")
        return self._dynamic_classifier

    async def _get_aesthetic_scorer(self):
        """Lazy load AestheticScorer."""
        if self._aesthetic_scorer is None and self._enable_cinematography:
            try:
                from core.processing.cinematography import AestheticScorer

                self._aesthetic_scorer = AestheticScorer(device=self._device)
                log.info("[DeepResearch] AestheticScorer loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] AestheticScorer failed: {e}")
        return self._aesthetic_scorer

    async def _get_technical_detector(self):
        """Lazy load TechnicalCueDetector."""
        if self._technical_detector is None and self._enable_cinematography:
            try:
                from core.processing.cinematography import TechnicalCueDetector

                self._technical_detector = TechnicalCueDetector()
                log.info("[DeepResearch] TechnicalCueDetector loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] TechnicalCueDetector failed: {e}")
        return self._technical_detector

    async def _get_languagebind(self):
        """Lazy load LanguageBindEncoder."""
        from config import settings

        if (
            self._languagebind is None
            and self._enable_video
            and getattr(settings, "enable_languagebind", False)
        ):
            try:
                from core.processing.video_understanding import (
                    LanguageBindEncoder,
                )

                self._languagebind = LanguageBindEncoder(device=self._device)
                log.info("[DeepResearch] LanguageBindEncoder loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] LanguageBindEncoder failed: {e}")
        return self._languagebind

    async def _get_internvideo(self):
        """Lazy load InternVideoEncoder."""
        if self._internvideo is None and self._enable_video:
            try:
                from core.processing.video_understanding import (
                    InternVideoEncoder,
                )

                self._internvideo = InternVideoEncoder(device=self._device)
                log.info("[DeepResearch] InternVideoEncoder loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] InternVideoEncoder failed: {e}")
        return self._internvideo

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    async def analyze_frame(
        self,
        frame: np.ndarray | Path | str,
        shot_concepts: list[str] | None = None,
        mood_concepts: list[str] | None = None,
        compute_embeddings: bool = True,
        compute_saliency: bool = False,
        compute_aesthetics: bool = True,
        compute_fingerprint: bool = False,
    ) -> DeepResearchResult:
        """Analyze a single frame with cinematography models.

        Args:
            frame: Frame as numpy array or path to image.
            shot_concepts: Custom shot type concepts for classification.
            mood_concepts: Custom mood concepts for classification.
            compute_aesthetics: Whether to compute aesthetic score.

        Returns:
            DeepResearchResult with cinematography data.
        """
        result = DeepResearchResult()

        # Load frame if path
        if isinstance(frame, (str, Path)):
            try:
                import cv2

                frame_arr = cv2.imread(str(frame))
                if frame_arr is None:
                    log.warning(f"[DeepResearch] Failed to load frame: {frame}")
                    return result
                frame = frame_arr
            except Exception as e:
                log.warning(f"[DeepResearch] Frame load error: {e}")
                return result

        # Default concepts for dynamic classification (Use Config)
        from config import settings

        if shot_concepts is None:
            shot_concepts = settings.cinematography_shot_types

        if mood_concepts is None:
            mood_concepts = settings.cinematography_moods

        # 1. Cinematography Analysis (shot type, mood)
        classifier = await self._get_dynamic_classifier()
        if classifier:
            try:
                # Shot type
                shot_results = await classifier.classify(
                    frame, shot_concepts, top_k=1
                )
                if shot_results:
                    result.shot_type = shot_results[0].get("concept", "")
                    result.shot_confidence = shot_results[0].get(
                        "confidence", 0.0
                    )

                # Mood
                mood_results = await classifier.classify(
                    frame, mood_concepts, top_k=1
                )
                if mood_results:
                    result.mood = mood_results[0].get("concept", "")
                    result.mood_confidence = mood_results[0].get(
                        "confidence", 0.0
                    )
            except Exception as e:
                log.warning(f"[DeepResearch] Classification failed: {e}")

        # 2. Aesthetic scoring
        if compute_aesthetics:
            scorer = await self._get_aesthetic_scorer()
            if scorer:
                try:
                    result.aesthetic_score = await scorer.score(frame)
                except Exception as e:
                    log.warning(f"[DeepResearch] Aesthetic scoring failed: {e}")

        # 3. Technical cues (black frame detection)
        detector = await self._get_technical_detector()
        if detector:
            try:
                result.is_black_frame = await detector.detect_black_frame(frame)
            except Exception as e:
                log.warning(f"[DeepResearch] Technical detection failed: {e}")

        return result

    async def analyze_video_segment(
        self,
        video_path: Path | str,
        start_time: float,
        end_time: float,
        sample_frames: int = 8,
    ) -> DeepResearchResult:
        """Analyze video segment with temporal models (InternVideo, LanguageBind).

        Args:
            video_path: Path to video file.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            sample_frames: Number of frames to sample.

        Returns:
            DeepResearchResult with video understanding data.
        """
        result = DeepResearchResult()

        # Extract frames for video models
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            duration = end_time - start_time
            frame_interval = duration / sample_frames

            frames = []
            for i in range(sample_frames):
                timestamp = start_time + (i * frame_interval)
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()

            if not frames:
                log.warning(
                    f"[DeepResearch] No frames extracted from {video_path}"
                )
                return result

        except Exception as e:
            log.warning(f"[DeepResearch] Frame extraction failed: {e}")
            return result

        # InternVideo2 for action recognition
        internvideo = await self._get_internvideo()
        if internvideo:
            try:
                action_result = await internvideo.recognize_action(frames)
                if action_result:
                    result.action_labels = action_result.get("actions", [])
                    if "features" in action_result:
                        result.video_features["internvideo"] = action_result[
                            "features"
                        ]
            except Exception as e:
                log.warning(f"[DeepResearch] InternVideo analysis failed: {e}")

        # LanguageBind for multimodal embedding
        languagebind = await self._get_languagebind()
        if languagebind:
            try:
                multimodal_emb = await languagebind.encode_video(frames)
                if multimodal_emb is not None:
                    result.video_features["languagebind"] = multimodal_emb
            except Exception as e:
                log.warning(f"[DeepResearch] LanguageBind encoding failed: {e}")

        # Shot boundary detection between frames
        detector = await self._get_technical_detector()
        if detector and len(frames) >= 2:
            try:
                result.is_shot_boundary = detector.detect_shot_boundary(
                    frames[0], frames[-1]
                )
            except Exception as e:
                log.debug(
                    f"[DeepResearch] Shot boundary detection skipped: {e}"
                )

        return result

    async def match_query_to_frame(
        self,
        query: str,
        frame: np.ndarray,
    ) -> float:
        """Match a text query to a frame using CLIP-based matching.

        Args:
            query: Text query.
            frame: Frame to match against.

        Returns:
            Similarity score (0-1).
        """
        classifier = await self._get_dynamic_classifier()
        if classifier:
            try:
                results = await classifier.match_query(frame, query)
                return results.get("score", 0.0)
            except Exception as e:
                log.warning(f"[DeepResearch] Query matching failed: {e}")
        return 0.0

    def cleanup(self) -> None:
        """Release all resources."""
        if self._dynamic_classifier:
            self._dynamic_classifier.cleanup()
        if self._aesthetic_scorer:
            self._aesthetic_scorer.cleanup()
        if self._languagebind:
            self._languagebind.cleanup()
        if self._internvideo:
            self._internvideo.cleanup()

        log.info("[DeepResearch] All resources released")


# Global processor instance
_deep_research_processor: DeepResearchProcessor | None = None


def get_deep_research_processor() -> DeepResearchProcessor:
    """Get or create the global DeepResearchProcessor."""
    global _deep_research_processor
    if _deep_research_processor is None:
        _deep_research_processor = DeepResearchProcessor()
    return _deep_research_processor
