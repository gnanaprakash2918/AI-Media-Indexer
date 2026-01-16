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

                    import torch
                    
                    self.processor = AutoImageProcessor.from_pretrained(
                        self.model_name
                    )
                    # Load in fp16 to reduce VRAM by ~50%
                    self.model = TimesformerForVideoClassification.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                    )

                    device = self._get_device()
                    self.model.to(device)
                    if device == "cuda":
                        self.model.half()  # Ensure fp16 on GPU
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


class TemporalMotionAnalyzer:
    """Frame-level motion analysis for temporal queries.
    
    Complements TimeSformer action recognition with low-level motion
    detection useful for queries like "pin wobbling 500-2000ms before falling".
    
    Usage:
        analyzer = TemporalMotionAnalyzer()
        result = analyzer.analyze_motion(frame1, frame2)
        # {"motion_intensity": "moderate", "motion_score": 32.5}
    """
    
    def __init__(
        self,
        motion_threshold_low: float = 10.0,
        motion_threshold_high: float = 50.0,
    ):
        """Initialize motion analyzer."""
        import cv2
        self.cv2 = cv2
        self.motion_threshold_low = motion_threshold_low
        self.motion_threshold_high = motion_threshold_high
    
    def analyze_motion(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> dict[str, Any]:
        """Analyze motion between two consecutive frames."""
        gray1 = self._to_gray(frame1)
        gray2 = self._to_gray(frame2)
        
        if gray1.shape != gray2.shape:
            gray1 = self.cv2.resize(gray1, (gray2.shape[1], gray2.shape[0]))
        
        diff = self.cv2.absdiff(gray1, gray2)
        motion_score = float(np.mean(diff))
        
        if motion_score < self.motion_threshold_low:
            intensity = "static"
        elif motion_score < self.motion_threshold_high:
            intensity = "moderate"
        else:
            intensity = "high"
        
        return {
            "motion_score": motion_score,
            "motion_intensity": intensity,
        }
    
    def detect_delayed_events(
        self,
        frames: list[np.ndarray],
        fps: float = 30.0,
        min_delay_ms: int = 500,
        max_delay_ms: int = 2000,
    ) -> list[dict[str, Any]]:
        """Detect events with specific delays (e.g., 'wobbling 500-2000ms')."""
        if len(frames) < 2:
            return []
        
        delayed_events = []
        frame_duration_ms = 1000 / fps
        min_frames = int(min_delay_ms / frame_duration_ms)
        max_frames = int(max_delay_ms / frame_duration_ms)
        
        moderate_start = None
        moderate_start_idx = None
        
        for i in range(1, len(frames)):
            motion = self.analyze_motion(frames[i-1], frames[i])
            
            if motion["motion_intensity"] == "moderate":
                if moderate_start is None:
                    moderate_start = i * frame_duration_ms
                    moderate_start_idx = i
            
            elif motion["motion_intensity"] == "high":
                if moderate_start is not None:
                    delay_frames = i - moderate_start_idx
                    if min_frames <= delay_frames <= max_frames:
                        delayed_events.append({
                            "type": "delayed_transition",
                            "start_ms": moderate_start,
                            "end_ms": i * frame_duration_ms,
                            "delay_ms": i * frame_duration_ms - moderate_start,
                        })
                moderate_start = None
                moderate_start_idx = None
            
            elif motion["motion_intensity"] == "static":
                moderate_start = None
                moderate_start_idx = None
        
        log.info(f"[Temporal] Found {len(delayed_events)} delayed events")
        return delayed_events
    
    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale."""
        if len(frame.shape) == 2:
            return frame
        return self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2GRAY)


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
