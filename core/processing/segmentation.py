"""Segment Anything Model 3 (SAM 3) Wrapper for Concept Tracking.

This module provides the interface for SAM 3's Promptable Concept Segmentation (PCS).
SAM 3 can track any concept from a text prompt across video frames with masklets.

Note: This is a standalone module. Integration into the main pipeline is optional
via the `enable_sam3_tracking` config flag.
"""

import gc
from pathlib import Path
from typing import Iterator, Any

import numpy as np
import torch

from config import settings
from core.utils.logger import log


class Sam3Tracker:
    """SAM 3 video segmentation and concept tracking."""
    
    def __init__(self):
        self.predictor = None
        self.inference_state = None
        self.device = settings.device
        self._initialized = False

    def initialize(self) -> bool:
        """Load SAM 3 model. Returns True if successful."""
        if self._initialized:
            return self.predictor is not None
            
        self._initialized = True
        
        try:
            from sam2.build_sam import build_sam2_video_predictor
            
            checkpoint = settings.model_cache_dir / "sam2" / "sam2_hiera_large.pt"
            config = "sam2_hiera_l.yaml"
            
            if not checkpoint.exists():
                log(f"[SAM3] Checkpoint not found: {checkpoint}")
                log("[SAM3] Download from: https://github.com/facebookresearch/segment-anything-2")
                return False
                
            log("[SAM3] Loading video predictor...")
            self.predictor = build_sam2_video_predictor(config, checkpoint)
            log(f"[SAM3] Loaded on {self.device}")
            return True
            
        except ImportError:
            log("[SAM3] sam2 package not installed. Run: pip install segment-anything-2")
            return False
        except Exception as e:
            log(f"[SAM3] Failed to load: {e}")
            return False

    def init_video(self, video_path: Path) -> bool:
        """Initialize video state for tracking."""
        if not self.initialize():
            return False
            
        try:
            self.inference_state = self.predictor.init_state(str(video_path))
            log(f"[SAM3] Initialized video: {video_path.name}")
            return True
        except Exception as e:
            log(f"[SAM3] Video init failed: {e}")
            return False

    def add_concept_prompt(self, text: str, frame_idx: int = 0) -> list[int]:
        """Add a text concept prompt for tracking.
        
        Args:
            text: Concept to track (e.g., "red car", "person").
            frame_idx: Frame to start tracking from.
            
        Returns:
            List of object IDs assigned to detected instances.
        """
        if self.predictor is None or self.inference_state is None:
            log("[SAM3] Not initialized")
            return []
            
        try:
            _, obj_ids, _ = self.predictor.add_new_prompt(
                self.inference_state,
                frame_idx=frame_idx,
                text=text
            )
            log(f"[SAM3] Added concept '{text}': {len(obj_ids)} instances")
            return obj_ids.tolist() if hasattr(obj_ids, 'tolist') else list(obj_ids)
        except Exception as e:
            log(f"[SAM3] Failed to add prompt: {e}")
            return []

    def propagate(self) -> Iterator[dict[str, Any]]:
        """Propagate masks through video frames.
        
        Yields:
            Dict with frame_idx, object_ids, and binary masks.
        """
        if self.predictor is None or self.inference_state is None:
            return
            
        try:
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                self.inference_state
            ):
                masks = (mask_logits > 0.0).cpu().numpy()
                yield {
                    "frame_idx": frame_idx,
                    "object_ids": obj_ids.tolist() if hasattr(obj_ids, 'tolist') else list(obj_ids),
                    "masks": masks,
                }
        except Exception as e:
            log(f"[SAM3] Propagation error: {e}")

    def process_video_concepts(
        self, 
        video_path: Path, 
        prompts: list[str]
    ) -> Iterator[dict[str, Any]]:
        """Full pipeline: init video, add prompts, propagate.
        
        Args:
            video_path: Path to video file.
            prompts: List of concepts to track.
            
        Yields:
            Masklet results per frame.
        """
        if not self.init_video(video_path):
            return
            
        for prompt in prompts:
            self.add_concept_prompt(prompt)
            
        yield from self.propagate()
        
        self.cleanup()

    def cleanup(self) -> None:
        """Release resources."""
        self.inference_state = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log("[SAM3] Resources released")

    def reset(self) -> None:
        """Full reset including model."""
        self.cleanup()
        self.predictor = None
        self._initialized = False
