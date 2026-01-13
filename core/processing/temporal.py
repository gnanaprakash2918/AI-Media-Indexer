"""Temporal understanding using TimeSformer for action recognition.

Enables queries like "when did he throw the ball" by extracting
action embeddings from video clips using TimeSformer.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class TemporalAnalyzer:
    """TimeSformer-based action recognition for temporal queries.

    Extracts action embeddings from video clips to enable
    temporal search like "throwing", "running", "dancing".

    Usage:
        analyzer = TemporalAnalyzer()
        actions = await analyzer.analyze_clip(frames)
        # [{"action": "throwing", "confidence": 0.85}]
    """

    def __init__(
        self,
        model_name: str = "facebook/timesformer-base-finetuned-k400",
        device: str | None = None,
    ):
        """Initialize temporal analyzer.

        Args:
            model_name: HuggingFace model name.
            device: Device to run on. Auto-detected if None.
        """
        self.model_name = model_name
        self._device = device
        self.model = None
        self.processor = None
        self._init_lock = asyncio.Lock()

    def _get_device(self) -> str:
        """Get device to use."""
        if self._device:
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    async def _lazy_load(self) -> bool:
        """Load TimeSformer model lazily.

        Returns:
            True if model loaded successfully.
        """
        if self.model is not None:
            return True

        async with self._init_lock:
            if self.model is not None:
                return True

            try:
                from core.utils.resource_arbiter import RESOURCE_ARBITER

                async with RESOURCE_ARBITER.acquire("timesformer", vram_gb=2.0):
                    log.info(f"[TimeSformer] Loading {self.model_name}")

                    from transformers import (
                        AutoImageProcessor,
                        TimesformerForVideoClassification,
                    )

                    self.processor = AutoImageProcessor.from_pretrained(
                        self.model_name
                    )
                    self.model = TimesformerForVideoClassification.from_pretrained(
                        self.model_name
                    )

                    device = self._get_device()
                    self.model.to(device)
                    self.model.eval()
                    self._device = device

                    log.info(f"[TimeSformer] Model loaded on {device}")
                    return True

            except ImportError as e:
                log.warning(f"[TimeSformer] transformers not available: {e}")
                return False
            except Exception as e:
                log.error(f"[TimeSformer] Failed to load: {e}")
                return False

    async def analyze_clip(
        self,
        frames: list[np.ndarray],
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Analyze a video clip for actions.

        Args:
            frames: List of RGB frames (8-16 frames recommended).
            top_k: Number of top predictions to return.
            threshold: Minimum confidence threshold.

        Returns:
            List of {"action": str, "confidence": float} dicts.
        """
        if not await self._lazy_load():
            return []

        if len(frames) < 4:
            log.warning("[TimeSformer] Need at least 4 frames")
            return []

        try:
            import torch
            from core.utils.resource_arbiter import RESOURCE_ARBITER

            async with RESOURCE_ARBITER.acquire("timesformer", vram_gb=2.0):
                # Sample 8 frames uniformly
                indices = np.linspace(
                    0, len(frames) - 1, num=8, dtype=int
                )
                sampled = [frames[i] for i in indices]

                # Process frames
                inputs = self.processor(
                    images=sampled,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

                # Get top-k predictions
                top_indices = probs.argsort()[::-1][:top_k]
                results = []
                for idx in top_indices:
                    conf = float(probs[idx])
                    if conf >= threshold:
                        label = self.model.config.id2label.get(idx, f"class_{idx}")
                        results.append({
                            "action": label,
                            "confidence": round(conf, 3),
                        })

                if results:
                    log.debug(f"[TimeSformer] Actions: {results}")

                return results

        except Exception as e:
            log.error(f"[TimeSformer] Analysis failed: {e}")
            return []

    async def get_action_embedding(
        self,
        frames: list[np.ndarray],
    ) -> np.ndarray | None:
        """Get action embedding for a video clip.

        Args:
            frames: List of RGB frames.

        Returns:
            Embedding as numpy array, or None if failed.
        """
        if not await self._lazy_load():
            return None

        if len(frames) < 4:
            return None

        try:
            import torch
            from core.utils.resource_arbiter import RESOURCE_ARBITER

            async with RESOURCE_ARBITER.acquire("timesformer", vram_gb=2.0):
                indices = np.linspace(0, len(frames) - 1, num=8, dtype=int)
                sampled = [frames[i] for i in indices]

                inputs = self.processor(
                    images=sampled,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    # Use last hidden state as embedding
                    hidden = outputs.hidden_states[-1]
                    # Pool over sequence
                    embedding = hidden.mean(dim=1).cpu().numpy()[0]

                return embedding

        except Exception as e:
            log.error(f"[TimeSformer] Embedding failed: {e}")
            return None

    def cleanup(self) -> None:
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        log.info("[TimeSformer] Resources released")


def extract_temporal_clips(
    video_path: str,
    clip_duration: float = 5.0,
    fps: int = 2,
) -> list[tuple[float, list[np.ndarray]]]:
    """Extract temporal clips from a video.

    Args:
        video_path: Path to video file.
        clip_duration: Duration of each clip in seconds.
        fps: Frames per second to extract.

    Yields:
        (start_time, frames) tuples.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"[Temporal] Cannot open video: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(video_fps / fps)
    frames_per_clip = int(clip_duration * fps)

    clips = []
    current_clip = []
    clip_start = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_clip.append(rgb)

            if len(current_clip) >= frames_per_clip:
                clips.append((clip_start, current_clip))
                clip_start = frame_count / video_fps
                current_clip = []

        frame_count += 1

    # Don't forget the last partial clip
    if len(current_clip) >= 4:
        clips.append((clip_start, current_clip))

    cap.release()

    log.info(
        f"[Temporal] Extracted {len(clips)} clips from {video_path}"
    )
    return clips
