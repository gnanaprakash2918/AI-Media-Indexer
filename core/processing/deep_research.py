"""Deep Research Integration Module.

Wires SOTA models into the ingestion pipeline:
- Video Understanding (LanguageBind) — gated by config

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


class DeepResearchProcessor:
    """Unified processor for Deep Research SOTA models.

    Usage:
        processor = DeepResearchProcessor()
        result = await processor.analyze_video_segment(video_path, start, end)
    """

    def __init__(
        self,
        enable_video_understanding: bool = True,
        enable_cinematography: bool = False,  # Kept for compatibility, but unused
        device: str | None = None,
    ):
        self._enable_video = enable_video_understanding
        self._device = device
        self._languagebind = None
        self._init_lock = asyncio.Lock()

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

    async def analyze_frame(
        self,
        frame: np.ndarray | Path | str,
        shot_concepts: list[str] | None = None,
        mood_concepts: list[str] | None = None,
        compute_embeddings: bool = True,
        compute_saliency: bool = False,
        compute_aesthetics: bool = False,
        compute_fingerprint: bool = False,
    ) -> DeepResearchResult:
        """Analyze a single frame. Cinematography models have been removed.
        Returns empty DeepResearchResult.
        """
        return DeepResearchResult()

    async def analyze_video_segment(
        self,
        video_path: Path | str,
        start_time: float,
        end_time: float,
        sample_frames: int = 8,
    ) -> DeepResearchResult:
        """Analyze video segment with temporal models (LanguageBind).

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

        # LanguageBind for multimodal embedding and action recognition
        languagebind = await self._get_languagebind()
        if languagebind:
            try:
                # 1. Action recognition (using LanguageBind as robust backend)
                common_actions = [
                    "running",
                    "walking",
                    "eating",
                    "drinking",
                    "talking",
                    "driving",
                    "dancing",
                    "cooking",
                    "fighting",
                    "playing sports",
                ]
                # Classify action using LanguageBind
                video_emb = await languagebind.encode_video(frames)
                if video_emb is not None:
                    result.video_features["languagebind"] = video_emb
                    
                    # Zero-shot action recognition
                    action_results = []
                    for label in common_actions:
                        text_emb = await languagebind.encode_text(f"a video of {label}")
                        if text_emb is not None:
                            similarity = float(np.dot(video_emb, text_emb))
                            action_results.append({
                                "action": label,
                                "confidence": round(similarity, 3)
                            })
                    action_results.sort(key=lambda x: x["confidence"], reverse=True)
                    result.action_labels = [a["action"] for a in action_results[:3]]

            except Exception as e:
                log.warning(f"[DeepResearch] LanguageBind analysis failed: {e}")

        return result

    async def match_query_to_frame(
        self,
        query: str,
        frame: np.ndarray,
    ) -> float:
        """Match a text query to a frame. Cinematography models have been removed.
        Returns 0.0.
        """
        return 0.0

    def cleanup(self) -> None:
        """Release all resources."""
        if self._languagebind:
            self._languagebind.cleanup()
        log.info("[DeepResearch] All resources released")


# Global processor instance
_deep_research_processor: DeepResearchProcessor | None = None


def get_deep_research_processor() -> DeepResearchProcessor:
    """Get or create the global DeepResearchProcessor."""
    global _deep_research_processor
    if _deep_research_processor is None:
        _deep_research_processor = DeepResearchProcessor()
    return _deep_research_processor
