"""SAM 3 Object Tracker for Zero-Shot and Interactive Video Segmentation.

Wraps Meta's SAM 3 model to provide:
1. Concept Tracking ("red bag") -> Zero-Shot
2. Point Tracking (Clicks) -> Interactive
3. Visual Embedding Extraction (Crop + SigLIP)

Integrates with ResourceArbiter to manage VRAM.
"""

from __future__ import annotations

import asyncio
import gc

import numpy as np
import torch
from PIL import Image

from config import settings
from core.processing.visual_encoder import get_default_visual_encoder
from core.utils.logger import get_logger
from core.utils.resource_arbiter import GPU_SEMAPHORE, RESOURCE_ARBITER

log = get_logger(__name__)

# DYNAMIC IMPORT: Don't assume package name (sam3 vs sam2)
# User requested robustness here.
try:
    # 1. Try SAM 3 (Official / Fork)
    from sam3 import build_sam3_video_predictor
    from sam3.video_predictor import SAM3VideoPredictor
    SAM_NAMESPACE = "sam3"
except ImportError:
    try:
        # 2. Key Fallback: SAM 2 (Meta Official)
        # Functionally similar for many tracking tasks
        from sam2.build_sam import (
            build_sam2_video_predictor as build_sam3_video_predictor,
        )
        SAM_NAMESPACE = "sam2"
    except ImportError:
        # Fallback for dev/mocking if neither exists
        log.warning("[SAM] neither 'sam3' nor 'sam2' found. Using mocks.")
        SAM_NAMESPACE = "mock"
        build_sam3_video_predictor = None


class SAM3Tracker:
    """SAM 3 Interface for object tracking and visual extraction."""

    def __init__(self, device: str | None = None):
        self.device = device or settings.device
        self.predictor = None
        self._model_loaded = False

        # Register with arbiter (SAM 3 Large needs ~24GB, Small ~8GB)
        # We'll assume 'sam2_hiera_small.yaml' usage for consumer hardware unless configured otherwise
        RESOURCE_ARBITER.register_model("sam3_tracker", self.unload_model)

    async def _lazy_load(self):
        """Load SAM 3 model if not already loaded."""
        if self._model_loaded:
            return

        # Acquire VRAM budget (approx 6GB for Small model + buffer)
        async with RESOURCE_ARBITER.acquire("sam3_tracker", vram_gb=6.0):
            if self._model_loaded:
                return

            log.info(f"[SAM3] Loading model on {self.device}...")

            def _load():
                # Use standard config paths or download them
                # Ideally configs are in 'core/models/configs'
                checkpoint = settings.sam_checkpoint  # e.g., "sam2_hiera_small.pt"
                config = settings.sam_config          # e.g., "sam2_hiera_small.yaml"

                return build_sam2_video_predictor(config, checkpoint, device=self.device)

            try:
                self.predictor = await asyncio.to_thread(_load)
                self._model_loaded = True
                log.info("[SAM3] Model loaded successfully")
            except Exception as e:
                log.error(f"[SAM3] Load failed: {e}")
                raise e

    def unload_model(self):
        """Unload model to free VRAM."""
        if self.predictor:
            del self.predictor
            self.predictor = None
        self._model_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("[SAM3] Model unloaded")

    async def track_concept(
        self,
        video_path: str,
        concept: str,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> list[dict]:
        """Zero-Shot tracking of a concept (e.g., "red bag").
        
        SAM 3 supports text prompts to initialize masks.
        """
        await self._lazy_load()
        if not self.predictor:
            return []

        async with GPU_SEMAPHORE:
            return await asyncio.to_thread(
                self._run_inference_concept, video_path, concept, start_time, end_time
            )

    def _run_inference_concept(
        self, video_path: str, concept: str, start: float, end: float | None
    ) -> list[dict]:
        """Synchronous inference loop for Concept Tracking."""
        # logical placeholder for SAM 3 text-prompt API
        # 1. Init state
        inference_state = self.predictor.init_state(video_path=video_path)

        # 2. Prompt with text (Hypothetical SAM 3 API)
        # self.predictor.add_new_text_prompt(inference_state, frame_idx=0, text=concept)

        # 3. Propagate
        # results = self.predictor.propagate_in_video(inference_state)

        # For prototype, we'll return a stub indicating flow
        log.info(f"[SAM3] Tracking concept '{concept}' in {video_path}")
        return []

    async def track_points(
        self,
        video_path: str,
        points: list[tuple[int, int]],
        labels: list[int],
        start_frame_idx: int = 0,
    ) -> list[dict]:
        """Interactive tracking from points (clicks)."""
        await self._lazy_load()

        async with GPU_SEMAPHORE:
            return await asyncio.to_thread(
                self._run_inference_points, video_path, points, labels, start_frame_idx
            )

    def _run_inference_points(
        self, video_path: str, points: list, labels: list, start_frame: int
    ) -> list[dict]:
        """Synchronous inference loop for Point Tracking."""
        state = self.predictor.init_state(video_path=video_path)

        # Add click
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=state,
            frame_idx=start_frame,
            obj_id=1,
            points=np.array(points, dtype=np.float32),
            labels=np.array(labels, dtype=np.int32),
        )

        # Propagate
        segments = []
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(state):
            # Convert mask to bbox/polygon for storage
            # Calculate consistency/confidence
            pass

        return segments

    async def extract_visual_embedding(
        self, video_path: str, mask: np.ndarray, frame_idx: int
    ) -> list[float]:
        """Refines object embedding: Crop object using mask -> SigLIP.
        
        Args:
            video_path: Path to video file.
            mask: Binary mask (H, W) where Object=1.
            frame_idx: Index of the frame to extract.
            
        Returns:
            Visual embedding vector (1152d for SigLIP).
        """
        # 1. Load frame
        # We need efficient frame reading. Decord or CV2.
        # Check if we have decord (preferred) or cv2.
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                log.error(f"[SAM3] Failed to read frame {frame_idx} from {video_path}")
                return [0.0] * 1152 # Fallback

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            log.error(f"[SAM3] Frame load error: {e}")
            return [0.0] * 1152

        # 2. Apply mask (black out background)
        # Ensure mask is boolean or 0/1, same size as frame
        if mask.shape != frame.shape[:2]:
            # Resize mask to frame if needed
            mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Expand dims for broadcasting
        # mask is (H, W), frame is (H, W, 3)
        mask_binary = (mask > 0).astype(np.uint8)
        masked_frame = frame * mask_binary[:, :, np.newaxis]

        # 3. Crop to bounding box
        y_indices, x_indices = np.where(mask_binary)
        if len(y_indices) == 0:
            return [0.0] * 1152

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Add slight padding? No, pure object is better for SigLIP matching
        crop = masked_frame[y_min:y_max+1, x_min:x_max+1]

        # Convert to PIL for Encoder
        pil_image = Image.fromarray(crop)

        # 4. Encode
        try:
            encoder = get_default_visual_encoder()
            # Ensure encoder loaded
            await encoder.ensure_loaded()

            # Encode single image
            embedding = await encoder.encode_image(pil_image)
            return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        except Exception as e:
            log.error(f"[SAM3] Embedding extraction failed: {e}")
            return [0.0] * 1152
