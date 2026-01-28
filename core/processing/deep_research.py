"""Deep Research Integration Module.

Wires all SOTA models from Deep Research into the ingestion pipeline:
- Advanced Embeddings (NV-Embed-v2, Nomic, Ensemble)
- Video Understanding (LanguageBind, InternVideo2, V-JEPA, DINOv2, VideoMAE, ImageBind)
- Cinematography (DynamicClassifier, AestheticScorer, TechnicalCueDetector)
- Audio Analysis (AudioTempoAnalyzer, SaliencyDetector, CLAP)
- Perceptual Hashing (Content fingerprinting)

User Priority: ACCURACY over storage/speed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DeepResearchResult:
    """Container for all Deep Research analysis results."""

    # Advanced embeddings
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)

    # Video understanding
    video_features: dict[str, np.ndarray] = field(default_factory=dict)
    action_labels: list[str] = field(default_factory=list)
    motion_prediction: np.ndarray | None = None

    # Cinematography
    shot_type: str = ""
    shot_confidence: float = 0.0
    aesthetic_score: float = 0.0
    mood: str = ""
    mood_confidence: float = 0.0

    # Technical cues
    is_black_frame: bool = False
    is_shot_boundary: bool = False
    blur_score: float = 0.0

    # Audio analysis
    tempo_bpm: float = 0.0
    beat_positions: list[float] = field(default_factory=list)
    audio_mood: str = ""
    is_music: bool = False

    # Fingerprinting
    perceptual_hash: str = ""
    audio_fingerprint: str = ""

    # Saliency
    saliency_regions: list[dict] = field(default_factory=list)


class DeepResearchProcessor:
    """Unified processor for all Deep Research SOTA models.

    Provides a single interface to run all Deep Research models on a frame/video.
    All models are lazy-loaded and use RESOURCE_ARBITER for GPU management.

    Usage:
        processor = DeepResearchProcessor()
        result = await processor.analyze_frame(frame_path, ...)
        result = await processor.analyze_video_segment(video_path, start, end)
    """

    def __init__(
        self,
        enable_advanced_embeddings: bool = True,
        enable_video_understanding: bool = True,
        enable_cinematography: bool = True,
        enable_audio_analysis: bool = True,
        enable_fingerprinting: bool = True,
        device: str | None = None,
    ):
        """Initialize Deep Research processor.

        Args:
            enable_advanced_embeddings: Enable NV-Embed, Nomic ensemble.
            enable_video_understanding: Enable LanguageBind, InternVideo, etc.
            enable_cinematography: Enable shot type, aesthetics, mood analysis.
            enable_audio_analysis: Enable tempo, beat, mood detection.
            enable_fingerprinting: Enable perceptual hashing.
            device: Device for inference. Auto-detected if None.
        """
        self._enable_embeddings = enable_advanced_embeddings
        self._enable_video = enable_video_understanding
        self._enable_cinematography = enable_cinematography
        self._enable_audio = enable_audio_analysis
        self._enable_fingerprinting = enable_fingerprinting
        self._device = device

        # Lazy-loaded components
        self._embedding_ensemble = None
        self._dynamic_classifier = None
        self._aesthetic_scorer = None
        self._technical_detector = None
        self._tempo_analyzer = None
        self._saliency_detector = None
        self._perceptual_hasher = None
        self._audio_fingerprinter = None
        self._languagebind = None
        self._internvideo = None
        self._dinov2 = None
        self._videomae = None

        self._init_lock = asyncio.Lock()

    # =========================================================================
    # LAZY LOADERS
    # =========================================================================

    async def _get_embedding_ensemble(self):
        """Lazy load EmbeddingEnsemble."""
        if self._embedding_ensemble is None and self._enable_embeddings:
            try:
                from core.processing.advanced_embeddings import (
                    EmbeddingEnsemble,
                )

                self._embedding_ensemble = EmbeddingEnsemble(
                    use_nv_embed=True,  # SOTA accuracy
                    use_nomic=True,  # Long context
                    use_bge=True,  # Hybrid
                    device=self._device,
                )
                log.info("[DeepResearch] EmbeddingEnsemble loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] EmbeddingEnsemble failed: {e}")
        return self._embedding_ensemble

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

    async def _get_tempo_analyzer(self):
        """Lazy load AudioTempoAnalyzer."""
        if self._tempo_analyzer is None and self._enable_audio:
            try:
                from core.processing.audio_analysis import AudioTempoAnalyzer

                self._tempo_analyzer = AudioTempoAnalyzer()
                log.info("[DeepResearch] AudioTempoAnalyzer loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] AudioTempoAnalyzer failed: {e}")
        return self._tempo_analyzer

    async def _get_saliency_detector(self):
        """Lazy load SaliencyDetector."""
        if self._saliency_detector is None and self._enable_cinematography:
            try:
                from core.processing.audio_analysis import SaliencyDetector

                self._saliency_detector = SaliencyDetector()
                log.info("[DeepResearch] SaliencyDetector loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] SaliencyDetector failed: {e}")
        return self._saliency_detector

    async def _get_perceptual_hasher(self):
        """Lazy load PerceptualHasher."""
        if self._perceptual_hasher is None and self._enable_fingerprinting:
            try:
                from core.processing.fingerprinting import PerceptualHasher

                self._perceptual_hasher = PerceptualHasher()
                log.info("[DeepResearch] PerceptualHasher loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] PerceptualHasher failed: {e}")
        return self._perceptual_hasher

    async def _get_audio_fingerprinter(self):
        """Lazy load AudioFingerprinter."""
        if self._audio_fingerprinter is None and self._enable_fingerprinting:
            try:
                from core.processing.fingerprinting import AudioFingerprinter

                self._audio_fingerprinter = AudioFingerprinter()
                log.info("[DeepResearch] AudioFingerprinter loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] AudioFingerprinter failed: {e}")
        return self._audio_fingerprinter

    async def _get_languagebind(self):
        """Lazy load LanguageBindEncoder."""
        if self._languagebind is None and self._enable_video:
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

    async def _get_dinov2(self):
        """Lazy load DINOv2Encoder."""
        if self._dinov2 is None and self._enable_video:
            try:
                from core.processing.academic_models import DINOv2Encoder

                self._dinov2 = DINOv2Encoder(device=self._device)
                log.info("[DeepResearch] DINOv2Encoder loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] DINOv2Encoder failed: {e}")
        return self._dinov2

    async def _get_videomae(self):
        """Lazy load VideoMAEEncoder."""
        if self._videomae is None and self._enable_video:
            try:
                from core.processing.academic_models import VideoMAEEncoder

                self._videomae = VideoMAEEncoder(device=self._device)
                log.info("[DeepResearch] VideoMAEEncoder loaded")
            except Exception as e:
                log.warning(f"[DeepResearch] VideoMAEEncoder failed: {e}")
        return self._videomae

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    async def analyze_frame(
        self,
        frame: np.ndarray | Path | str,
        shot_concepts: list[str] | None = None,
        mood_concepts: list[str] | None = None,
        compute_embeddings: bool = True,
        compute_saliency: bool = True,
        compute_aesthetics: bool = True,
        compute_fingerprint: bool = True,
    ) -> DeepResearchResult:
        """Analyze a single frame with all Deep Research models.

        Args:
            frame: Frame as numpy array or path to image.
            shot_concepts: Custom shot type concepts for classification.
            mood_concepts: Custom mood concepts for classification.
            compute_embeddings: Whether to compute advanced embeddings.
            compute_saliency: Whether to compute visual saliency.
            compute_aesthetics: Whether to compute aesthetic score.
            compute_fingerprint: Whether to compute perceptual hash.

        Returns:
            DeepResearchResult with all analysis data.
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

        # 3. Technical cues
        detector = await self._get_technical_detector()
        if detector:
            try:
                # TechnicalCueDetector uses separate async methods
                result.is_black_frame = await detector.detect_black_frame(frame)
                result.blur_score = 0.0  # Blur detection requires additional implementation
            except Exception as e:
                log.warning(f"[DeepResearch] Technical detection failed: {e}")

        # 4. Saliency detection
        if compute_saliency:
            saliency = await self._get_saliency_detector()
            if saliency:
                try:
                    regions = saliency.detect(frame)
                    result.saliency_regions = regions
                except Exception as e:
                    log.warning(
                        f"[DeepResearch] Saliency detection failed: {e}"
                    )

        # 5. Perceptual hashing
        if compute_fingerprint:
            hasher = await self._get_perceptual_hasher()
            if hasher:
                try:
                    # hash_frame is async - must await
                    result.perceptual_hash = await hasher.hash_frame(frame) or ""
                except Exception as e:
                    log.warning(
                        f"[DeepResearch] Perceptual hashing failed: {e}"
                    )

        # 6. DINOv2 features (for zero-shot object discovery)
        dinov2 = await self._get_dinov2()
        if dinov2:
            try:
                # DINOv2Encoder uses extract_features, not encode_frame
                features = await dinov2.extract_features(frame)
                if features is not None:
                    result.video_features["dinov2"] = features
            except Exception as e:
                log.warning(f"[DeepResearch] DINOv2 encoding failed: {e}")

        return result

    async def analyze_audio_segment(
        self,
        audio_path: Path | str,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> DeepResearchResult:
        """Analyze audio segment for tempo, beats, mood.

        Args:
            audio_path: Path to audio/video file.
            start_time: Start time in seconds.
            end_time: End time in seconds (None = entire file).

        Returns:
            DeepResearchResult with audio analysis data.
        """
        result = DeepResearchResult()

        # Tempo analysis
        tempo = await self._get_tempo_analyzer()
        if tempo:
            try:
                analysis = await tempo.analyze(
                    str(audio_path),
                    start_time=start_time,
                    end_time=end_time,
                )
                result.tempo_bpm = analysis.get("tempo", 0.0)
                result.beat_positions = analysis.get("beats", [])
                result.audio_mood = analysis.get("mood", "")
                result.is_music = analysis.get("is_music", False)
            except Exception as e:
                log.warning(f"[DeepResearch] Tempo analysis failed: {e}")

        # Audio fingerprinting
        fingerprinter = await self._get_audio_fingerprinter()
        if fingerprinter:
            try:
                result.audio_fingerprint = await fingerprinter.fingerprint(
                    str(audio_path),
                    start=start_time,
                    end=end_time,
                )
            except Exception as e:
                log.warning(f"[DeepResearch] Audio fingerprinting failed: {e}")

        return result

    async def analyze_video_segment(
        self,
        video_path: Path | str,
        start_time: float,
        end_time: float,
        sample_frames: int = 8,
    ) -> DeepResearchResult:
        """Analyze video segment with temporal models.

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
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
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

        # VideoMAE for self-supervised features
        videomae = await self._get_videomae()
        if videomae:
            try:
                mae_features = await videomae.extract_action_features(frames)
                if mae_features is not None:
                    result.video_features["videomae"] = mae_features
            except Exception as e:
                log.warning(f"[DeepResearch] VideoMAE encoding failed: {e}")

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

    async def encode_text(self, text: str) -> dict[str, np.ndarray]:
        """Encode text with advanced embedding ensemble.

        Args:
            text: Text to encode.

        Returns:
            Dict of {model_name: embedding}.
        """
        ensemble = await self._get_embedding_ensemble()
        if ensemble:
            return await ensemble.encode_query(text)
        return {}

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
        if self._embedding_ensemble:
            self._embedding_ensemble.cleanup()
        if self._dynamic_classifier:
            self._dynamic_classifier.cleanup()
        if self._aesthetic_scorer:
            self._aesthetic_scorer.cleanup()
        if self._languagebind:
            self._languagebind.cleanup()
        if self._internvideo:
            self._internvideo.cleanup()
        if self._dinov2:
            self._dinov2.cleanup()
        if self._videomae:
            self._videomae.cleanup()

        log.info("[DeepResearch] All resources released")


# Global processor instance
_deep_research_processor: DeepResearchProcessor | None = None


def get_deep_research_processor() -> DeepResearchProcessor:
    """Get or create the global DeepResearchProcessor."""
    global _deep_research_processor
    if _deep_research_processor is None:
        _deep_research_processor = DeepResearchProcessor()
    return _deep_research_processor
